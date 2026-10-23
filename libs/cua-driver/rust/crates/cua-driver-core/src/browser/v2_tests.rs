//! End-to-end tests for the v2 DOM-ref slice against a deterministic
//! mock CDP endpoint: composed shadow DOM, same-process iframes,
//! capability-tested OOPIFs with containment, frame/document identity
//! revalidation, navigation invalidation, and unproven-capability
//! omission/refusal.

use std::collections::HashSet;
use std::io::Cursor;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use serde_json::{json, Value};

use crate::action_record::ActionExecutionRecord;
use crate::protocol::{Content, ToolResult};
use crate::tool::Tool;

use super::engine::BrowserEngine;
use super::mock_cdp::{MockCdpServer, MockEvent, MockHandler, MockReply};
use super::platform::{
    BrowserConsentOutcome, BrowserConsentRequest, BrowserPlatform, ExistingProfileSetupOutcome,
    ExistingProfileSetupRequest, PrepareAction, PrepareOutcome, PrepareRequest,
};
use super::pointer::BrowserPointerTool;
use super::refusal::BrowserRefusal;
use super::select::BrowserSelectTool;
use super::tools::{
    browser_protected_resource_scope, BrowserClickTool, BrowserNavigateTool, BrowserPrepareTool,
    BrowserSetInputFilesTool, BrowserTypeTool, GetBrowserStateTool,
};
use super::types::{
    BrowserClassification, BrowserEngineFamily, BrowserProcessRole, BrowserProduct,
    EndpointOwnershipMethod, EndpointOwnershipProof, EndpointTransport, NativeOwnershipMethod,
    NativeOwnershipProof, NativeWindowInfo, OwnedEndpoint, ProcessFingerprint, Rect,
};

// ── Scripted Chromium fixture ────────────────────────────────────────────────

/// Mutable knobs + a call log, shared between the mock handler and the
/// test body. Tests flip loaders/capabilities mid-run to simulate
/// navigations, frame removal, and capability regression.
#[derive(Debug)]
struct FixtureState {
    oopif_supported: bool,
    oopif_present: bool,
    emit_rogue_attach: bool,
    main_url: String,
    main_loader: String,
    iframe_loader: String,
    oopif_loader: String,
    tab_sessions: u64,
    oopif_sessions: u64,
    fail_key_down_after: Option<usize>,
    completed_key_pairs: usize,
    semantic_large_page: bool,
    semantic_full_dom_fails: bool,
    semantic_full_dom_times_out: bool,
    semantic_truncated_dom: bool,
    screenshot_data: String,
    viewport_css_width: f64,
    viewport_css_height: f64,
    tab_visible: bool,
    /// Extra page targets Chromium reports but cannot map to any browser
    /// window (`Browser.getWindowForTarget` → `-32000 Browser window not
    /// found`): an extension side panel, an offscreen document.
    windowless_targets: Vec<Value>,
    /// Drop the primary tab target T1 from `Target.getTargets`.
    omit_primary_target: bool,
    /// Make `Browser.getWindowForTarget` for T1 answer this CDP error.
    primary_window_error: Option<(i64, String)>,
    /// A file chooser the page requests when a control gets user
    /// activation: `(backendNodeId, mode)`; no backend id models a File
    /// System Access picker.
    chooser_on_activation: Option<(Option<i64>, &'static str)>,
    /// Sessions with the Page domain enabled / chooser interception on.
    page_enabled: HashSet<String>,
    intercepting: HashSet<String>,
    /// Choosers that reached the desktop because nothing intercepted them.
    native_pickers_shown: usize,
    /// What the related-file-input probe finds for any ref: directly
    /// (the ref is, labels, contains, or controls it) or only nearby.
    related_field_direct: bool,
    related_field_nearby: bool,
    related_field_multiple: bool,
    /// A download the page starts on activation: (suggested name, bytes),
    /// written under an opaque id into the approved `downloadPath`.
    download_on_activation: Option<(&'static str, &'static [u8])>,
    download_path: Option<String>,
    /// A form with a native country <select> and a "Next" button, and what
    /// the page answers to the native and ARIA select functions.
    select_page: bool,
    native_select_reply: Value,
    /// Árvores AX de um dropdown feito à mão, uma por clique entregue.
    aria_trees: Vec<Value>,
    aria_stage: usize,
    /// Every incoming CDP call: (sessionId, method, params).
    calls: Vec<(Option<String>, String, Value)>,
}

impl Default for FixtureState {
    fn default() -> Self {
        Self {
            oopif_supported: true,
            oopif_present: true,
            emit_rogue_attach: false,
            main_url: "https://fixture.test/".into(),
            windowless_targets: Vec::new(),
            omit_primary_target: false,
            primary_window_error: None,
            main_loader: "L_MAIN_1".into(),
            iframe_loader: "L_IFRAME_1".into(),
            oopif_loader: "L_OOPIF_1".into(),
            tab_sessions: 0,
            oopif_sessions: 0,
            fail_key_down_after: None,
            completed_key_pairs: 0,
            semantic_large_page: false,
            semantic_full_dom_fails: false,
            semantic_full_dom_times_out: false,
            semantic_truncated_dom: false,
            screenshot_data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZJrAAAAAASUVORK5CYII=".into(),
            viewport_css_width: 800.0,
            viewport_css_height: 600.0,
            tab_visible: true,
            chooser_on_activation: None,
            page_enabled: HashSet::new(),
            intercepting: HashSet::new(),
            native_pickers_shown: 0,
            related_field_direct: false,
            related_field_nearby: false,
            related_field_multiple: false,
            download_on_activation: None,
            download_path: None,
            select_page: false,
            native_select_reply: Value::Null,
            aria_trees: Vec::new(),
            aria_stage: 0,
            calls: Vec::new(),
        }
    }
}

fn screenshot_png_base64(width: u32, height: u32) -> String {
    let image = RgbaImage::from_pixel(width, height, Rgba([18, 171, 52, 255]));
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image)
        .write_to(&mut encoded, ImageFormat::Png)
        .expect("encode screenshot fixture PNG");
    BASE64.encode(encoded.into_inner())
}

type SharedState = Arc<StdMutex<FixtureState>>;

/// Main-frame document: a plain button, an open shadow root with an
/// input, a user-agent shadow root (must be skipped), a same-process
/// iframe with a button, and an OOPIF placeholder (no contentDocument).
fn main_document() -> Value {
    json!({
        "root": {
            "nodeType": 9,
            "nodeName": "#document",
            "documentURL": "https://fixture.test/",
            "frameId": "F_MAIN",
            "children": [{
                "nodeType": 1,
                "nodeName": "HTML",
                "backendNodeId": 1,
                "children": [
                    {
                        "nodeType": 1,
                        "nodeName": "BUTTON",
                        "backendNodeId": 10,
                        "attributes": ["id", "main-btn"],
                    },
                    {
                        "nodeType": 1,
                        "nodeName": "DIV",
                        "backendNodeId": 11,
                        "shadowRoots": [{
                            "nodeType": 11,
                            "nodeName": "#document-fragment",
                            "shadowRootType": "open",
                            "backendNodeId": 12,
                            "children": [{
                                "nodeType": 1,
                                "nodeName": "INPUT",
                                "backendNodeId": 20,
                                "attributes": ["aria-label", "Shadow Input", "type", "text"],
                            }]
                        }]
                    },
                    {
                        "nodeType": 1,
                        "nodeName": "INPUT",
                        "backendNodeId": 21,
                        "attributes": ["type", "text"],
                        "shadowRoots": [{
                            "nodeType": 11,
                            "nodeName": "#document-fragment",
                            "shadowRootType": "user-agent",
                            "backendNodeId": 23,
                            "children": [{
                                "nodeType": 1,
                                "nodeName": "DIV",
                                "backendNodeId": 22,
                                "attributes": ["role", "button"],
                            }]
                        }]
                    },
                    {
                        "nodeType": 1,
                        "nodeName": "IFRAME",
                        "backendNodeId": 13,
                        "frameId": "F_IFRAME",
                        "contentDocument": {
                            "nodeType": 9,
                            "nodeName": "#document",
                            "frameId": "F_IFRAME",
                            "children": [{
                                "nodeType": 1,
                                "nodeName": "BUTTON",
                                "backendNodeId": 30,
                                "attributes": ["id", "inner-btn"],
                            }]
                        }
                    },
                    {
                        "nodeType": 1,
                        "nodeName": "IFRAME",
                        "backendNodeId": 14,
                        "frameId": "F_OOPIF",
                    }
                ]
            }]
        }
    })
}

/// Large application document used to prove that hidden retained controls do
/// not consume the semantic snapshot budget ahead of the active view.
fn large_semantic_document() -> Value {
    let mut children = Vec::new();
    for id in 0..320_i64 {
        children.push(json!({
            "nodeType": 1,
            "nodeName": "BUTTON",
            "backendNodeId": 1_000 + id,
            "attributes": [
                "aria-hidden", "true",
                "style", "display:none",
                "aria-label", format!("Retained control {id}"),
            ],
        }));
    }
    children.extend([
        json!({
            "nodeType": 1,
            "nodeName": "H1",
            "backendNodeId": 2_000,
            "children": [{
                "nodeType": 3,
                "nodeName": "#text",
                "nodeValue": "Visible message",
                "backendNodeId": 2_001,
            }],
        }),
        json!({
            "nodeType": 1,
            "nodeName": "P",
            "backendNodeId": 2_002,
            "children": [{
                "nodeType": 3,
                "nodeName": "#text",
                "nodeValue": "Please review the attached fixture report.",
                "backendNodeId": 2_003,
            }],
        }),
        json!({
            "nodeType": 1,
            "nodeName": "TEXTAREA",
            "backendNodeId": 2_010,
            "attributes": ["aria-label", "Reply body"],
        }),
        json!({
            "nodeType": 1,
            "nodeName": "BUTTON",
            "backendNodeId": 2_011,
            "attributes": ["aria-label", "Reply"],
        }),
    ]);
    for id in 0..305_i64 {
        children.push(json!({
            "nodeType": 1,
            "nodeName": "BUTTON",
            "backendNodeId": 3_000 + id,
            "attributes": ["aria-label", format!("Archive item {id}")],
        }));
    }
    json!({
        "root": {
            "nodeType": 9,
            "nodeName": "#document",
            "documentURL": "https://fixture.test/inbox/item-2",
            "frameId": "F_MAIN",
            "children": [{
                "nodeType": 1,
                "nodeName": "HTML",
                "backendNodeId": 999,
                "children": children,
            }]
        }
    })
}

fn truncated_semantic_document() -> Value {
    json!({
        "root": {
            "nodeType": 9,
            "nodeName": "#document",
            "documentURL": "https://fixture.test/inbox/item-2",
            "frameId": "F_MAIN",
            "children": [{
                "nodeType": 1,
                "nodeName": "HTML",
                "backendNodeId": 999,
                "childNodeCount": 629,
                "children": [],
            }]
        }
    })
}

fn large_semantic_ax_tree(frame_id: &str) -> Value {
    if frame_id != "F_MAIN" {
        return json!({"nodes": [{
            "nodeId": format!("root-{frame_id}"),
            "ignored": false,
            "role": {"value": "RootWebArea"},
            "name": {"value": "Inner fixture"},
            "childIds": []
        }]});
    }
    let mut child_ids = vec![
        "heading".to_owned(),
        "body".to_owned(),
        "editor".to_owned(),
        "reply".to_owned(),
    ];
    child_ids.extend((0..305).map(|id| format!("archive-{id}")));
    let mut nodes = vec![
        json!({
            "nodeId": "root-main",
            "ignored": false,
            "role": {"value": "RootWebArea"},
            "name": {"value": "Fixture inbox"},
            "childIds": child_ids
        }),
        json!({
            "nodeId": "heading",
            "parentId": "root-main",
            "ignored": false,
            "backendDOMNodeId": 2000,
            "role": {"value": "heading"},
            "name": {"value": "Visible message"},
            "childIds": []
        }),
        json!({
            "nodeId": "body",
            "parentId": "root-main",
            "ignored": false,
            "backendDOMNodeId": 2003,
            "role": {"value": "StaticText"},
            "name": {"value": "Please review the attached fixture report."},
            "childIds": []
        }),
        json!({
            "nodeId": "editor",
            "parentId": "root-main",
            "ignored": false,
            "backendDOMNodeId": 2010,
            "role": {"value": "textbox"},
            "name": {"value": "Reply body"},
            "value": {"value": ""},
            "properties": [
                {"name": "editable", "value": {"value": "plaintext"}},
                {"name": "focusable", "value": {"value": true}}
            ],
            "childIds": []
        }),
        json!({
            "nodeId": "reply",
            "parentId": "root-main",
            "ignored": false,
            "backendDOMNodeId": 2011,
            "role": {"value": "button"},
            "name": {"value": "Reply"},
            "properties": [
                {"name": "disabled", "value": {"value": false}},
                {"name": "focusable", "value": {"value": true}}
            ],
            "childIds": []
        }),
    ];
    for id in 0..305_i64 {
        nodes.push(json!({
            "nodeId": format!("archive-{id}"),
            "parentId": "root-main",
            "ignored": false,
            "backendDOMNodeId": 3_000 + id,
            "role": {"value": "button"},
            "name": {"value": format!("Archive item {id}")},
            "childIds": []
        }));
    }
    json!({"nodes": nodes})
}

fn semantic_layout_snapshot(backends: &[i64], bounds: &[[f64; 4]]) -> Value {
    let styles = (0..backends.len())
        .map(|_| json!([0, 1, 2, 3, 3]))
        .collect::<Vec<_>>();
    let node_indices = (0..backends.len()).collect::<Vec<_>>();
    let paint_orders = (1..=backends.len()).collect::<Vec<_>>();
    json!({
        "strings": ["block", "visible", "1", "auto", "pointer"],
        "documents": [{
            "nodes": {"backendNodeId": backends},
            "layout": {
                "nodeIndex": node_indices,
                "bounds": bounds,
                "styles": styles,
                "paintOrders": paint_orders
            }
        }]
    })
}

fn select_document() -> Value {
    let option = |backend: i64, label: &str| {
        json!({"nodeType": 1, "nodeName": "OPTION", "backendNodeId": backend,
               "children": [{"nodeType": 3, "nodeName": "#text", "nodeValue": label}]})
    };
    json!({
        "root": {
            "nodeType": 9,
            "nodeName": "#document",
            "documentURL": "https://fixture.test/",
            "frameId": "F_MAIN",
            "children": [{
                "nodeType": 1,
                "nodeName": "HTML",
                "backendNodeId": 1,
                "children": [
                    {"nodeType": 1, "nodeName": "SELECT", "backendNodeId": 40,
                     "children": [option(41, "Selecionar"), option(42, "Países Baixos"), option(43, "Portugal")]},
                    {"nodeType": 1, "nodeName": "BUTTON", "backendNodeId": 44,
                     "attributes": ["id", "next"]}
                ]
            }]
        }
    })
}

fn oopif_document() -> Value {
    json!({
        "root": {
            "nodeType": 9,
            "nodeName": "#document",
            "documentURL": "https://ads.example/frame",
            "frameId": "F_OOPIF",
            "children": [{
                "nodeType": 1,
                "nodeName": "HTML",
                "backendNodeId": 90,
                "children": [{
                    "nodeType": 1,
                    "nodeName": "INPUT",
                    "backendNodeId": 100,
                    "attributes": ["id", "ad-input", "type", "text"],
                }]
            }]
        }
    })
}

fn fixture_handler(state: SharedState) -> MockHandler {
    Arc::new(move |call| {
        let mut st = state.lock().unwrap();
        st.calls.push((
            call.session_id.clone(),
            call.method.clone(),
            call.params.clone(),
        ));
        let sess = call.session_id.clone().unwrap_or_default();
        let is_tab = sess.starts_with("tab-sess-");
        let is_oopif = sess.starts_with("oopif-sess-");

        // Chromium intercepts a chooser only for a session that has both the
        // Page domain and interception on; otherwise the native picker shows.
        let request_chooser = |st: &mut FixtureState| -> Vec<MockEvent> {
            let Some((backend, mode)) = st.chooser_on_activation else {
                return Vec::new();
            };
            let mut catching: Vec<&String> = st
                .intercepting
                .iter()
                .filter(|session| {
                    st.page_enabled.contains(*session) && session.starts_with("tab-sess-")
                })
                .collect();
            catching.sort();
            match catching.first() {
                Some(session) => {
                    let mut params = json!({ "frameId": "F_MAIN", "mode": mode });
                    if let Some(backend) = backend {
                        params["backendNodeId"] = json!(backend);
                    }
                    vec![MockEvent {
                        method: "Page.fileChooserOpened".into(),
                        session_id: Some((*session).clone()),
                        params,
                    }]
                }
                None => {
                    st.native_pickers_shown += 1;
                    Vec::new()
                }
            }
        };

        match call.method.as_str() {
            "Page.enable" => {
                st.page_enabled.insert(sess.clone());
                MockReply::ok(json!({}))
            }
            "Page.disable" => {
                st.page_enabled.remove(&sess);
                MockReply::ok(json!({}))
            }
            "Page.setInterceptFileChooserDialog" => {
                if call.params["enabled"].as_bool() == Some(true) {
                    st.intercepting.insert(sess.clone());
                } else {
                    st.intercepting.remove(&sess);
                }
                MockReply::ok(json!({}))
            }
            "Target.detachFromTarget" => {
                let detached = call.params["sessionId"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned();
                st.intercepting.remove(&detached);
                st.page_enabled.remove(&detached);
                MockReply::ok(json!({}))
            }
            "DOM.setFileInputFiles" | "Input.dispatchDragEvent" => MockReply::ok(json!({})),
            "Input.dispatchMouseEvent" if call.params["type"] == "mouseReleased" => {
                st.aria_stage += 1;
                let events = request_chooser(&mut st);
                MockReply::ok(json!({})).with_events(events)
            }
            "Browser.setDownloadBehavior" => {
                st.download_path = call.params["downloadPath"].as_str().map(str::to_owned);
                MockReply::ok(json!({}))
            }
            "Runtime.callFunctionOn" if call.params["userGesture"] == true => {
                st.aria_stage += 1;
                let mut events = request_chooser(&mut st);
                if let (Some((name, bytes)), Some(path)) =
                    (st.download_on_activation, st.download_path.clone())
                {
                    let guid = "6a8843d3-2cf2-4723-849c-37e48debd4a9";
                    std::fs::write(std::path::Path::new(&path).join(guid), bytes).unwrap();
                    events.push(MockEvent {
                        method: "Browser.downloadWillBegin".into(),
                        session_id: None,
                        params: json!({
                            "frameId": "F_MAIN", "guid": guid,
                            "url": "https://fixture.test/report", "suggestedFilename": name
                        }),
                    });
                    events.push(MockEvent {
                        method: "Browser.downloadProgress".into(),
                        session_id: None,
                        params: json!({ "guid": guid, "state": "completed", "receivedBytes": bytes.len() }),
                    });
                }
                MockReply::ok(json!({ "result": { "type": "undefined" } })).with_events(events)
            }
            "Runtime.callFunctionOn"
                if call.params["functionDeclaration"]
                    .as_str()
                    .is_some_and(|f| f.contains("const isFile")) =>
            {
                let nearby = call.params["arguments"][0]["value"] == true;
                if st.related_field_direct || (nearby && st.related_field_nearby) {
                    MockReply::ok(json!({
                        "result": { "type": "object", "subtype": "node", "objectId": "related-field" }
                    }))
                } else {
                    MockReply::ok(
                        json!({ "result": { "type": "object", "subtype": "null", "value": null } }),
                    )
                }
            }
            "DOM.describeNode" if call.params["objectId"] == "related-field" => {
                let mut attributes = vec![json!("type"), json!("file")];
                if st.related_field_multiple {
                    attributes.extend([json!("multiple"), json!("")]);
                }
                MockReply::ok(json!({
                    "node": { "nodeName": "INPUT", "backendNodeId": 77, "attributes": attributes }
                }))
            }
            "Target.getTargets" => {
                let mut infos: Vec<Value> = Vec::new();
                if !st.omit_primary_target {
                    infos.push(json!({
                        "targetId": "T1",
                        "type": "page",
                        "title": "Fixture",
                        "url": "https://fixture.test/",
                        "attached": false,
                    }));
                }
                infos.extend(st.windowless_targets.iter().cloned());
                MockReply::ok(json!({ "targetInfos": infos }))
            }
            "Target.getTargetInfo" => MockReply::ok(json!({
                "targetInfo": {
                    "targetId": "T1",
                    "type": "page",
                    "title": "Fixture after navigation",
                    "url": "https://fixture.test/",
                    "attached": true,
                }
            })),
            "Browser.getWindowForTarget" => {
                let target_id = call.params["targetId"].as_str().unwrap_or("");
                let windowless = st
                    .windowless_targets
                    .iter()
                    .any(|t| t["targetId"].as_str() == Some(target_id));
                if windowless {
                    MockReply::err(-32000, "Browser window not found")
                } else if let Some((code, message)) = st.primary_window_error.clone() {
                    MockReply::err(code, &message)
                } else {
                    MockReply::ok(json!({ "windowId": 11 }))
                }
            }
            "Browser.getWindowBounds" => MockReply::ok(json!({
                "bounds": { "left": 0.0, "top": 0.0, "width": 800.0, "height": 600.0 }
            })),
            "Target.attachToTarget" => {
                st.tab_sessions += 1;
                MockReply::ok(json!({ "sessionId": format!("tab-sess-{}", st.tab_sessions) }))
            }
            "Page.getFrameTree" if is_tab => MockReply::ok(json!({
                "frameTree": {
                    "frame": {
                        "id": "F_MAIN",
                        "loaderId": st.main_loader.clone(),
                        "url": st.main_url.clone(),
                    },
                    "childFrames": [{
                        "frame": {
                            "id": "F_IFRAME",
                            "parentId": "F_MAIN",
                            "loaderId": st.iframe_loader.clone(),
                            "url": "https://fixture.test/inner",
                        }
                    }]
                }
            })),
            "Page.getFrameTree" if is_oopif => MockReply::ok(json!({
                "frameTree": {
                    "frame": {
                        "id": "F_OOPIF",
                        "loaderId": st.oopif_loader.clone(),
                        "url": "https://ads.example/frame",
                    }
                }
            })),
            "DOM.getDocument" if is_tab && st.select_page => MockReply::ok(select_document()),
            "Accessibility.getFullAXTree" if is_tab && !st.aria_trees.is_empty() => {
                let stage = st.aria_stage.min(st.aria_trees.len() - 1);
                MockReply::ok(st.aria_trees[stage].clone())
            }
            "Accessibility.getFullAXTree" if is_tab && st.select_page => {
                MockReply::ok(json!({"nodes": [
                    {"nodeId": "root", "ignored": false, "role": {"value": "RootWebArea"},
                     "childIds": ["country", "next"]},
                    {"nodeId": "country", "parentId": "root", "ignored": false,
                     "backendDOMNodeId": 40, "role": {"value": "combobox"},
                     "name": {"value": "País ou região"}, "value": {"value": "Selecionar"},
                     "properties": [{"name": "expanded", "value": {"value": false}}]},
                    {"nodeId": "next", "parentId": "root", "ignored": false,
                     "backendDOMNodeId": 44, "role": {"value": "button"},
                     "name": {"value": "Seguinte"}}
                ]}))
            }
            "Runtime.callFunctionOn"
                if call.params["functionDeclaration"]
                    .as_str()
                    .is_some_and(|f| f.contains("const isSelect")) =>
            {
                MockReply::ok(
                    json!({ "result": { "type": "object", "value": st.native_select_reply.clone() } }),
                )
            }
            "DOM.getDocument" if is_tab => {
                let depth = call.params["depth"].as_i64().unwrap_or(-1);
                if st.semantic_full_dom_times_out && (depth == -1 || depth > 8) {
                    MockReply::err(-32000, "CDP DOM.getDocument timed out after 20s")
                } else if st.semantic_full_dom_fails && (depth == -1 || depth > 8) {
                    MockReply::err(-32000, "Object reference chain is too long")
                } else if st.semantic_truncated_dom && depth == 8 {
                    MockReply::ok(truncated_semantic_document())
                } else {
                    MockReply::ok(if st.semantic_large_page {
                        large_semantic_document()
                    } else {
                        main_document()
                    })
                }
            }
            "DOM.describeNode" if is_tab && call.params["backendNodeId"] == 999 => {
                MockReply::ok(json!({
                    "node": large_semantic_document()["root"]["children"][0].clone()
                }))
            }
            "DOM.getDocument" if is_oopif => MockReply::ok(oopif_document()),
            "Accessibility.getFullAXTree" if is_tab => {
                let frame_id = call.params["frameId"].as_str().unwrap_or("F_MAIN");
                if st.semantic_large_page {
                    MockReply::ok(large_semantic_ax_tree(frame_id))
                } else {
                    MockReply::ok(json!({"nodes": []}))
                }
            }
            "Accessibility.getFullAXTree" if is_oopif => MockReply::ok(json!({"nodes": [
                {"nodeId": "oopif-root", "ignored": false,
                 "role": {"value": "RootWebArea"}, "childIds": ["oopif-input"]},
                {"nodeId": "oopif-input", "parentId": "oopif-root", "ignored": false,
                 "backendDOMNodeId": 100, "role": {"value": "textbox"},
                 "name": {"value": "Embedded input"},
                 "properties": [{"name": "editable", "value": {"value": "plaintext"}}],
                 "childIds": []}
            ]})),
            "DOMSnapshot.captureSnapshot" if is_tab => {
                if st.semantic_large_page {
                    let mut backends = vec![999, 2000, 2003, 2010, 2011];
                    let mut bounds = vec![
                        [0.0, 0.0, 800.0, 600.0],
                        [20.0, 20.0, 500.0, 40.0],
                        [20.0, 80.0, 600.0, 80.0],
                        [20.0, 180.0, 600.0, 120.0],
                        [20.0, 320.0, 100.0, 36.0],
                    ];
                    for id in 0..305_i64 {
                        backends.push(3_000 + id);
                        bounds.push([20.0, 2_000.0 + id as f64 * 40.0, 160.0, 30.0]);
                    }
                    MockReply::ok(semantic_layout_snapshot(&backends, &bounds))
                } else {
                    MockReply::ok(semantic_layout_snapshot(&[], &[]))
                }
            }
            "DOMSnapshot.captureSnapshot" if is_oopif => MockReply::ok(semantic_layout_snapshot(
                &[90, 100],
                &[[0.0, 0.0, 300.0, 100.0], [10.0, 10.0, 120.0, 30.0]],
            )),
            "Page.getLayoutMetrics" => MockReply::ok(json!({
                "cssVisualViewport": {
                    "pageX": 0.0,
                    "pageY": 0.0,
                    "clientWidth": st.viewport_css_width,
                    "clientHeight": st.viewport_css_height
                }
            })),
            "Runtime.evaluate"
                if call.params["expression"] == "document.visibilityState === 'visible'" =>
            {
                MockReply::ok(json!({
                    "result": {
                        "type": "boolean",
                        "value": st.tab_visible
                    }
                }))
            }
            "Page.captureScreenshot" if is_tab => {
                MockReply::ok(json!({"data": st.screenshot_data.clone()}))
            }
            "Page.navigate" if is_tab => MockReply::ok(json!({
                "frameId": "F_MAIN",
                "loaderId": "L_MAIN_NAVIGATED",
            })),
            "Target.setAutoAttach" if is_tab => {
                if call.params["autoAttach"].as_bool() == Some(false) {
                    return MockReply::ok(json!({}));
                }
                if !st.oopif_supported {
                    return MockReply::method_not_found("Target.setAutoAttach");
                }
                let mut events = Vec::new();
                if st.oopif_present {
                    st.oopif_sessions += 1;
                    events.push(MockEvent {
                        method: "Target.attachedToTarget".into(),
                        session_id: Some(sess.clone()),
                        params: json!({
                            "sessionId": format!("oopif-sess-{}", st.oopif_sessions),
                            "targetInfo": {
                                "targetId": "T_OOPIF",
                                "type": "iframe",
                                "url": "https://ads.example/frame",
                                "attached": true,
                            },
                            "waitingForDebugger": false,
                        }),
                    });
                }
                if st.emit_rogue_attach {
                    // A child announced on a session we never proved.
                    events.push(MockEvent {
                        method: "Target.attachedToTarget".into(),
                        session_id: Some("unproven-sess".into()),
                        params: json!({
                            "sessionId": "rogue-sess",
                            "targetInfo": {
                                "targetId": "T_ROGUE",
                                "type": "iframe",
                                "url": "https://evil.example/",
                                "attached": true,
                            },
                            "waitingForDebugger": false,
                        }),
                    });
                    // A popup page target beneath the tab — wrong type.
                    events.push(MockEvent {
                        method: "Target.attachedToTarget".into(),
                        session_id: Some(sess.clone()),
                        params: json!({
                            "sessionId": "popup-sess",
                            "targetInfo": {
                                "targetId": "T_POPUP",
                                "type": "page",
                                "url": "https://fixture.test/popup",
                                "attached": true,
                            },
                            "waitingForDebugger": false,
                        }),
                    });
                }
                MockReply::ok(json!({})).with_events(events)
            }
            "DOM.scrollIntoViewIfNeeded" => MockReply::ok(json!({})),
            "DOM.getBoxModel" => {
                let backend = call.params["backendNodeId"].as_i64().unwrap_or(0);
                let known = if is_oopif {
                    backend == 100
                } else {
                    [10, 20, 21, 30].contains(&backend) || st.select_page
                };
                if known {
                    let (x, y) = ((backend * 10) as f64, (backend * 10) as f64);
                    MockReply::ok(json!({
                        "model": {
                            "content": [x, y, x + 20.0, y, x + 20.0, y + 10.0, x, y + 10.0]
                        }
                    }))
                } else {
                    MockReply::err(-32000, "No node with given id")
                }
            }
            "Input.dispatchKeyEvent" => {
                let event_type = call.params["type"].as_str().unwrap_or_default();
                if event_type == "keyDown" && st.fail_key_down_after == Some(st.completed_key_pairs)
                {
                    MockReply::err(-32000, "fixture key delivery failure")
                } else {
                    if event_type == "keyUp" {
                        st.completed_key_pairs += 1;
                    }
                    MockReply::ok(json!({}))
                }
            }
            "DOM.focus"
            | "Emulation.setFocusEmulationEnabled"
            | "Input.dispatchMouseEvent"
            | "Input.insertText" => MockReply::ok(json!({})),
            "DOM.resolveNode" => MockReply::ok(json!({
                "object": { "objectId": format!("obj-{}", call.params["backendNodeId"]) }
            })),
            "Runtime.callFunctionOn" => MockReply::ok(json!({ "result": { "value": true } })),
            other => MockReply::method_not_found(other),
        }
    })
}

/// Platform adapter pointing at the fixture endpoint: pid 1 is a
/// Chromium browser owning native window 7 at (0,0,800,600).
struct FixturePlatform {
    ws_url: String,
    trusted_input_limited: bool,
    managed_endpoint_visible: bool,
    process_role: BrowserProcessRole,
    managed_discovery_invoked: Arc<AtomicBool>,
    existing_endpoint_visible: Arc<AtomicBool>,
    setup_invoked: Arc<AtomicBool>,
    setup_aborted: Arc<AtomicBool>,
    stall_consent: bool,
}

#[async_trait]
impl BrowserPlatform for FixturePlatform {
    fn standalone_trusted_input_background_limitation(&self) -> Option<&'static str> {
        self.trusted_input_limited
            .then_some("fixture trusted input raises the standalone window")
    }

    async fn classify_browser(&self, _pid: i64) -> Result<BrowserClassification, BrowserRefusal> {
        Ok(BrowserClassification {
            is_browser: true,
            engine: BrowserEngineFamily::Chromium,
            product_kind: BrowserProduct::GoogleChrome,
            product: Some("MockChrome".into()),
            channel: Some("stable".into()),
            // The mock endpoint is an explicit in-process harness, not a
            // personal standalone browser profile.
            process_role: self.process_role,
            supports_cdp: true,
        })
    }

    async fn native_window(
        &self,
        pid: i64,
        window_id: u64,
    ) -> Result<NativeWindowInfo, BrowserRefusal> {
        Ok(NativeWindowInfo {
            pid,
            window_id,
            title: "Fixture - Chrome".into(),
            bounds: Rect::new(0.0, 0.0, 800.0, 600.0),
            geometry_exact: true,
            ownership: NativeOwnershipProof {
                method: NativeOwnershipMethod::WindowServerOwner,
                owner_pid: pid,
                detail: None,
            },
        })
    }

    async fn is_only_exact_native_window(
        &self,
        _pid: i64,
        _window_id: u64,
    ) -> Result<Option<bool>, BrowserRefusal> {
        Ok(Some(true))
    }

    async fn discover_owned_endpoint(
        &self,
        pid: i64,
    ) -> Result<Option<OwnedEndpoint>, BrowserRefusal> {
        self.managed_discovery_invoked.store(true, Ordering::SeqCst);
        if !self.managed_endpoint_visible {
            return Ok(None);
        }
        self.discover_existing_profile_endpoint(pid).await
    }

    async fn discover_existing_profile_endpoint(
        &self,
        pid: i64,
    ) -> Result<Option<OwnedEndpoint>, BrowserRefusal> {
        if !self.existing_endpoint_visible.load(Ordering::SeqCst) {
            return Ok(None);
        }
        Ok(Some(OwnedEndpoint {
            ws_url: self.ws_url.clone(),
            http_port: None,
            transport: EndpointTransport::LegacyJsonVersion,
            ownership: EndpointOwnershipProof {
                method: EndpointOwnershipMethod::ListeningSocketPid,
                owner_pid: pid,
                listener_pid: None,
                detail: None,
            },
        }))
    }

    async fn reprove_existing_profile_endpoint(
        &self,
        pid: i64,
        expected_ws_url: &str,
    ) -> Result<Option<OwnedEndpoint>, BrowserRefusal> {
        if expected_ws_url != self.ws_url {
            return Ok(None);
        }
        Ok(Some(OwnedEndpoint {
            ws_url: self.ws_url.clone(),
            http_port: None,
            transport: EndpointTransport::LegacyJsonVersion,
            ownership: EndpointOwnershipProof {
                method: EndpointOwnershipMethod::ListeningSocketPid,
                owner_pid: pid,
                listener_pid: None,
                detail: Some("fixture exact approved endpoint".to_owned()),
            },
        }))
    }

    async fn setup_existing_profile_endpoint(
        &self,
        _request: ExistingProfileSetupRequest,
    ) -> Result<ExistingProfileSetupOutcome, BrowserRefusal> {
        self.setup_invoked.store(true, Ordering::SeqCst);
        Ok(ExistingProfileSetupOutcome {
            opened_setup_page: true,
            closed_setup_page: false,
            enabled_remote_debugging: true,
            used_bounded_pixel_fallback: false,
            focused_setup_address_field: true,
            foregrounded_window: false,
            injected_global_input: false,
            endpoint: Some(OwnedEndpoint {
                ws_url: self.ws_url.clone(),
                http_port: None,
                transport: EndpointTransport::DevToolsActivePort,
                ownership: EndpointOwnershipProof {
                    method: EndpointOwnershipMethod::ListeningSocketPid,
                    owner_pid: 1,
                    listener_pid: None,
                    detail: Some("fixture exact setup transition".to_owned()),
                },
            }),
        })
    }

    async fn commit_existing_profile_setup(
        &self,
        _request: ExistingProfileSetupRequest,
    ) -> Result<bool, BrowserRefusal> {
        Ok(true)
    }

    async fn abort_existing_profile_setup(
        &self,
        _request: ExistingProfileSetupRequest,
        error: BrowserRefusal,
    ) -> BrowserRefusal {
        self.setup_aborted.store(true, Ordering::SeqCst);
        error
    }

    async fn handle_existing_profile_consent(
        &self,
        _request: BrowserConsentRequest,
    ) -> Result<BrowserConsentOutcome, BrowserRefusal> {
        if self.stall_consent {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            return Ok(BrowserConsentOutcome::NotPresent);
        }
        Err(BrowserRefusal::new(
            super::refusal::BrowserRefusalCode::BrowserRouteUnavailable,
            "fixture consent prompt is unavailable",
        ))
    }

    async fn process_fingerprint(&self, pid: i64) -> Result<ProcessFingerprint, BrowserRefusal> {
        Ok(ProcessFingerprint {
            pid,
            start_time: Some(1),
            executable: None,
        })
    }

    async fn prepare_endpoint(
        &self,
        _request: PrepareRequest,
    ) -> Result<PrepareOutcome, BrowserRefusal> {
        Ok(PrepareOutcome {
            action: PrepareAction::NoOp,
            endpoint: None,
            message: "fixture: nothing to do".into(),
            prepared_pid: None,
            side_effects: Default::default(),
            attachment: None,
        })
    }
}

// ── Scaffolding ──────────────────────────────────────────────────────────────

struct Fixture {
    state: SharedState,
    // Kept alive for the test's duration; dropping it kills the endpoint.
    _server: MockCdpServer,
    engine: Arc<BrowserEngine>,
    setup_invoked: Arc<AtomicBool>,
}

async fn fixture_with(configure: impl FnOnce(&mut FixtureState)) -> Fixture {
    fixture_with_platform(configure, false).await
}

async fn fixture_with_platform(
    configure: impl FnOnce(&mut FixtureState),
    trusted_input_limited: bool,
) -> Fixture {
    let mut initial = FixtureState::default();
    configure(&mut initial);
    let state = Arc::new(StdMutex::new(initial));
    let server = MockCdpServer::start(fixture_handler(state.clone())).await;
    let setup_invoked = Arc::new(AtomicBool::new(false));
    let engine = BrowserEngine::new(Arc::new(FixturePlatform {
        ws_url: server.ws_url(),
        trusted_input_limited,
        managed_endpoint_visible: true,
        process_role: BrowserProcessRole::EmbeddedApplication,
        managed_discovery_invoked: Arc::new(AtomicBool::new(false)),
        existing_endpoint_visible: Arc::new(AtomicBool::new(true)),
        setup_invoked: setup_invoked.clone(),
        setup_aborted: Arc::new(AtomicBool::new(false)),
        stall_consent: false,
    }));
    Fixture {
        state,
        _server: server,
        engine,
        setup_invoked,
    }
}

async fn fixture() -> Fixture {
    fixture_with(|_| {}).await
}

struct FixtureProtectedProvider {
    consent_seen: AtomicBool,
}

#[async_trait]
impl crate::consent::ProtectedConsentProvider for FixtureProtectedProvider {
    fn provider_id(&self) -> &'static str {
        "test.browser-protected-provider"
    }

    async fn request_consent(
        &self,
        request: &crate::consent::ConsentRequest,
    ) -> Result<crate::consent::ProviderDecision, String> {
        self.consent_seen.store(true, Ordering::SeqCst);
        Ok(crate::consent::ProviderDecision {
            action: crate::consent::ConsentAction::Accept,
            request_digest: request.request_digest.clone(),
        })
    }
}

async fn protected_existing_profile_fixture() -> (Fixture, Arc<FixtureProtectedProvider>) {
    let state = Arc::new(StdMutex::new(FixtureState::default()));
    let server = MockCdpServer::start(fixture_handler(state.clone())).await;
    let setup_invoked = Arc::new(AtomicBool::new(false));
    let provider = Arc::new(FixtureProtectedProvider {
        consent_seen: AtomicBool::new(false),
    });
    let engine = BrowserEngine::new_with_protected_consent_provider(
        Arc::new(FixturePlatform {
            ws_url: server.ws_url(),
            trusted_input_limited: false,
            managed_endpoint_visible: false,
            process_role: BrowserProcessRole::StandaloneConsumer,
            managed_discovery_invoked: Arc::new(AtomicBool::new(false)),
            existing_endpoint_visible: Arc::new(AtomicBool::new(true)),
            setup_invoked: setup_invoked.clone(),
            setup_aborted: Arc::new(AtomicBool::new(false)),
            stall_consent: false,
        }),
        Some(provider.clone()),
    );
    (
        Fixture {
            state,
            _server: server,
            engine,
            setup_invoked,
        },
        provider,
    )
}

async fn existing_profile_setup_fixture() -> (Fixture, Arc<AtomicBool>) {
    let state = Arc::new(StdMutex::new(FixtureState::default()));
    let server = MockCdpServer::start(fixture_handler(state.clone())).await;
    let setup_invoked = Arc::new(AtomicBool::new(false));
    let engine = BrowserEngine::new_with_protected_consent_provider(
        Arc::new(FixturePlatform {
            ws_url: server.ws_url(),
            trusted_input_limited: false,
            managed_endpoint_visible: false,
            process_role: BrowserProcessRole::StandaloneConsumer,
            managed_discovery_invoked: Arc::new(AtomicBool::new(false)),
            existing_endpoint_visible: Arc::new(AtomicBool::new(false)),
            setup_invoked: setup_invoked.clone(),
            setup_aborted: Arc::new(AtomicBool::new(false)),
            stall_consent: false,
        }),
        Some(Arc::new(FixtureProtectedProvider {
            consent_seen: AtomicBool::new(false),
        })),
    );
    (
        Fixture {
            state,
            _server: server,
            engine,
            setup_invoked: setup_invoked.clone(),
        },
        setup_invoked,
    )
}

fn structured(result: &ToolResult) -> &Value {
    result
        .structured_content
        .as_ref()
        .expect("structured content")
}

const SESSION: &str = "run-v2";

async fn bind(f: &Fixture) -> (String, String) {
    let tool = GetBrowserStateTool::new(f.engine.clone());
    let result = tool
        .invoke(json!({ "pid": 1, "window_id": 7, "session": SESSION }))
        .await;
    let s = structured(&result);
    assert_eq!(s["status"], "ok", "bind must succeed: {s}");
    assert_eq!(s["binding_quality"], "exact");
    let target_id = s["target_id"].as_str().unwrap().to_owned();
    let tab_id = s["tabs"][0]["tab_id"].as_str().unwrap().to_owned();
    (target_id, tab_id)
}

#[tokio::test]
async fn standalone_consumer_bind_without_grant_refuses_before_endpoint_discovery() {
    let state = Arc::new(StdMutex::new(FixtureState::default()));
    let server = MockCdpServer::start(fixture_handler(state)).await;
    let managed_discovery_invoked = Arc::new(AtomicBool::new(false));
    let setup_invoked = Arc::new(AtomicBool::new(false));
    let engine = BrowserEngine::new(Arc::new(FixturePlatform {
        ws_url: server.ws_url(),
        trusted_input_limited: false,
        managed_endpoint_visible: true,
        process_role: BrowserProcessRole::StandaloneConsumer,
        managed_discovery_invoked: managed_discovery_invoked.clone(),
        existing_endpoint_visible: Arc::new(AtomicBool::new(true)),
        setup_invoked: setup_invoked.clone(),
        setup_aborted: Arc::new(AtomicBool::new(false)),
        stall_consent: false,
    }));

    let result = GetBrowserStateTool::new(engine)
        .invoke(json!({ "pid": 1, "window_id": 7, "session": SESSION }))
        .await;
    let refusal = structured(&result);
    assert_eq!(refusal["status"], "refused");
    assert_eq!(refusal["refusal"]["code"], "browser_consent_required");
    assert_eq!(
        refusal["refusal"]["detail"]["reason"],
        "consumer_profile_endpoint_requires_grant"
    );
    assert_eq!(
        refusal["refusal"]["detail"]["next_action"],
        "browser_prepare"
    );
    assert!(
        !managed_discovery_invoked.load(Ordering::SeqCst),
        "read-only bind must not inspect a consent-gated endpoint"
    );
    assert!(
        !setup_invoked.load(Ordering::SeqCst),
        "read-only bind must never invoke browser setup UI"
    );
}

#[tokio::test]
async fn approved_existing_profile_attach_claims_then_binds_one_generation() {
    // Match Chrome's per-instance toggle: the endpoint is discoverable only
    // through the approved existing-profile route, never as driver-managed.
    let (f, provider) = protected_existing_profile_fixture().await;
    let prepare = BrowserPrepareTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": SESSION,
            "_transport_session_id": "transport-v2-attach",
            "strategy": { "kind": "existing_profile" }
        }))
        .await;
    let prepared = structured(&prepare);
    assert_eq!(prepared["status"], "ok", "{prepared}");
    assert_eq!(prepared["action"], "attached_existing_profile");
    assert_eq!(prepared["attachment"]["kind"], "existing_profile");
    assert_eq!(prepared["attachment"]["capabilities_invalidated"], true);
    assert_eq!(prepared["side_effects"]["displayed_consent_prompt"], false);
    assert!(provider.consent_seen.load(Ordering::SeqCst));
    assert!(!f.setup_invoked.load(Ordering::SeqCst));

    let state = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": SESSION,
            "_transport_session_id": "transport-v2-attach"
        }))
        .await;
    assert_eq!(structured(&state)["status"], "ok", "{}", structured(&state));
    crate::session::fire_session_end("transport-v2-attach");
}

#[tokio::test]
async fn approved_existing_profile_tools_stay_within_the_reviewed_cdp_surface() {
    const TRANSPORT: &str = "transport-v2-method-policy";
    let (f, _) = protected_existing_profile_fixture().await;
    let prepare = BrowserPrepareTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": SESSION,
            "_transport_session_id": TRANSPORT,
            "strategy": { "kind": "existing_profile" }
        }))
        .await;
    assert_eq!(
        structured(&prepare)["status"],
        "ok",
        "{}",
        structured(&prepare)
    );

    let state = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": SESSION,
            "_transport_session_id": TRANSPORT
        }))
        .await;
    let bound = structured(&state);
    assert_eq!(bound["status"], "ok", "{bound}");
    let target = bound["target_id"].as_str().unwrap().to_owned();
    let tab = bound["tabs"][0]["tab_id"].as_str().unwrap().to_owned();

    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let input = ref_of(&snap, "main", "Shadow Input");
    let clicked = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "ref": button,
            "input_route": "dom_event",
            "session": SESSION
        }))
        .await;
    assert_eq!(
        structured(&clicked)["status"],
        "ok",
        "{}",
        structured(&clicked)
    );
    let typed = BrowserTypeTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "ref": input,
            "text": "policy-check",
            "session": SESSION
        }))
        .await;
    assert_eq!(structured(&typed)["status"], "ok", "{}", structured(&typed));
    let navigated = BrowserNavigateTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "url": "https://fixture.test/policy-check",
            "session": SESSION
        }))
        .await;
    assert_eq!(
        structured(&navigated)["status"],
        "ok",
        "{}",
        structured(&navigated)
    );

    let forbidden = [
        "Runtime.enable",
        "Target.setDiscoverTargets",
        "Page.addScriptToEvaluateOnNewDocument",
        "Network.enable",
        "Fetch.enable",
        "Emulation.setUserAgentOverride",
        "Emulation.setDeviceMetricsOverride",
        "Emulation.setTimezoneOverride",
    ];
    let calls = f.state.lock().unwrap().calls.clone();
    for method in forbidden {
        assert!(
            calls.iter().all(|(_, observed, _)| observed != method),
            "existing-profile operation emitted forbidden CDP method {method}: {calls:?}"
        );
    }

    crate::session::fire_session_end(TRANSPORT);
}

#[tokio::test]
async fn protected_provider_accepts_exact_attach_and_session_end_revokes_the_grant() {
    const PROTECTED_SESSION: &str = "protected-provider-v2";
    const PROTECTED_TRANSPORT: &str = "protected-transport-v2";
    let (f, provider) = protected_existing_profile_fixture().await;
    let prepare = BrowserPrepareTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": PROTECTED_SESSION,
            "_transport_session_id": PROTECTED_TRANSPORT,
            "strategy": { "kind": "existing_profile" }
        }))
        .await;
    assert_eq!(
        structured(&prepare)["status"],
        "ok",
        "{}",
        structured(&prepare)
    );
    assert!(provider.consent_seen.load(Ordering::SeqCst));

    crate::session::fire_session_end(PROTECTED_TRANSPORT);
    tokio::task::yield_now().await;

    let state = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": PROTECTED_SESSION,
            "_transport_session_id": PROTECTED_TRANSPORT
        }))
        .await;
    assert_eq!(structured(&state)["status"], "refused");
}

#[tokio::test]
async fn approved_existing_profile_setup_reports_exact_side_effects() {
    let (f, setup_invoked) = existing_profile_setup_fixture().await;
    let prepare = BrowserPrepareTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": SESSION,
            "strategy": { "kind": "existing_profile" }
        }))
        .await;
    let prepared = structured(&prepare);
    assert_eq!(prepared["status"], "ok", "{prepared}");
    assert!(setup_invoked.load(Ordering::SeqCst));
    assert_eq!(prepared["side_effects"]["opened_setup_page"], true);
    assert_eq!(prepared["side_effects"]["closed_setup_page"], true);
    assert_eq!(prepared["side_effects"]["enabled_remote_debugging"], true);
    assert_eq!(prepared["side_effects"]["changed_preferences"], true);
    assert_eq!(
        prepared["side_effects"]["focused_setup_address_field"],
        true
    );

    let state = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": SESSION
        }))
        .await;
    assert_eq!(structured(&state)["status"], "ok", "{}", structured(&state));
}

#[tokio::test]
async fn refused_consent_cancels_stalled_claim_before_revoking_grant() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let stalled_server = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    });
    let engine = BrowserEngine::new_with_protected_consent_provider(
        Arc::new(FixturePlatform {
            ws_url: format!("ws://{address}/devtools/browser"),
            trusted_input_limited: false,
            managed_endpoint_visible: false,
            process_role: BrowserProcessRole::StandaloneConsumer,
            managed_discovery_invoked: Arc::new(AtomicBool::new(false)),
            existing_endpoint_visible: Arc::new(AtomicBool::new(false)),
            setup_invoked: Arc::new(AtomicBool::new(false)),
            setup_aborted: Arc::new(AtomicBool::new(false)),
            stall_consent: false,
        }),
        Some(Arc::new(FixtureProtectedProvider {
            consent_seen: AtomicBool::new(false),
        })),
    );
    let prepared = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        BrowserPrepareTool::new(engine).invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": SESSION,
            "strategy": { "kind": "existing_profile" }
        })),
    )
    .await
    .expect("a refused consent route must not deadlock grant revocation");
    assert_eq!(structured(&prepared)["status"], "refused");
    assert_eq!(
        structured(&prepared)["refusal"]["code"],
        "browser_route_unavailable"
    );
    stalled_server.abort();
}

#[tokio::test]
async fn cancelled_prepare_aborts_the_exact_pending_setup() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let stalled_server = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    });
    let setup_aborted = Arc::new(AtomicBool::new(false));
    let engine = BrowserEngine::new_with_protected_consent_provider(
        Arc::new(FixturePlatform {
            ws_url: format!("ws://{address}/devtools/browser"),
            trusted_input_limited: false,
            managed_endpoint_visible: false,
            process_role: BrowserProcessRole::StandaloneConsumer,
            managed_discovery_invoked: Arc::new(AtomicBool::new(false)),
            existing_endpoint_visible: Arc::new(AtomicBool::new(false)),
            setup_invoked: Arc::new(AtomicBool::new(false)),
            setup_aborted: setup_aborted.clone(),
            stall_consent: true,
        }),
        Some(Arc::new(FixtureProtectedProvider {
            consent_seen: AtomicBool::new(false),
        })),
    );
    let cancelled = tokio::time::timeout(
        std::time::Duration::from_millis(750),
        BrowserPrepareTool::new(engine).invoke(json!({
            "pid": 1,
            "window_id": 7,
            "session": SESSION,
            "strategy": { "kind": "existing_profile" }
        })),
    )
    .await;
    assert!(
        cancelled.is_err(),
        "fixture must cancel during consent handling"
    );
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !setup_aborted.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping prepare must schedule exact pending-setup rollback");
    stalled_server.abort();
}

async fn snapshot(f: &Fixture, target_id: &str, tab_id: &str) -> Value {
    let tool = GetBrowserStateTool::new(f.engine.clone());
    let result = tool
        .invoke(json!({ "target_id": target_id, "tab_id": tab_id, "session": SESSION }))
        .await;
    structured(&result).clone()
}

async fn semantic_snapshot(f: &Fixture, target_id: &str, tab_id: &str) -> Value {
    let tool = GetBrowserStateTool::new(f.engine.clone());
    let result = tool
        .invoke(json!({
            "target_id": target_id,
            "tab_id": tab_id,
            "session": SESSION,
            "snapshot_format": "semantic_v2"
        }))
        .await;
    structured(&result).clone()
}

async fn semantic_snapshot_with(f: &Fixture, target_id: &str, tab_id: &str, extra: Value) -> Value {
    let mut args = json!({
        "target_id": target_id,
        "tab_id": tab_id,
        "session": SESSION,
        "snapshot_format": "semantic_v2"
    });
    args.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let result = GetBrowserStateTool::new(f.engine.clone())
        .invoke(args)
        .await;
    structured(&result).clone()
}

/// The `ref` string of the first snapshot entry in the given frame kind
/// with the given backing label fragment (or any, if empty).
fn ref_of(snapshot: &Value, frame: &str, label_fragment: &str) -> String {
    snapshot["refs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| {
            r["frame"] == frame
                && (label_fragment.is_empty()
                    || r["label"].as_str().unwrap_or("").contains(label_fragment))
        })
        .unwrap_or_else(|| panic!("no {frame} ref with label ~{label_fragment:?}: {snapshot}"))
        ["ref"]
        .as_str()
        .unwrap()
        .to_owned()
}

fn recorded_calls(f: &Fixture, method: &str) -> Vec<(Option<String>, Value)> {
    f.state
        .lock()
        .unwrap()
        .calls
        .iter()
        .filter(|(_, m, _)| m == method)
        .map(|(s, _, p)| (s.clone(), p.clone()))
        .collect()
}

// ── Snapshot composition ─────────────────────────────────────────────────────

#[tokio::test]
async fn snapshot_composes_shadow_iframe_and_oopif_refs() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;

    assert_eq!(snap["status"], "ok", "{snap}");
    let frames: Vec<&str> = snap["refs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["frame"].as_str().unwrap())
        .collect();
    assert_eq!(
        frames,
        vec!["main", "main", "main", "iframe", "oopif"],
        "main button + shadow input + plain input, iframe button, oopif input: {snap}"
    );
    assert_eq!(snap["oopif"]["status"], "attached");
    assert_eq!(snap["oopif"]["frames"], 1);
    assert_eq!(snap["truncated"], false);

    // Composed shadow content keeps its labels; user-agent shadow
    // internals (backend 22, role=button) must not have been minted.
    let labels: Vec<&str> = snap["refs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["label"].as_str().unwrap_or(""))
        .collect();
    assert!(labels.iter().any(|l| l.contains("Shadow Input")), "{snap}");
    assert!(
        !labels.iter().any(|l| l.contains("role=button")),
        "user-agent shadow content leaked: {snap}"
    );
}

#[tokio::test]
async fn semantic_snapshot_refreshes_bind_time_title_from_main_document() {
    let f = fixture_with(|st| st.semantic_large_page = true).await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot(&f, &target, &tab).await;

    assert_eq!(snap["status"], "ok", "{snap}");
    assert_eq!(snap["page"]["title"], "Fixture inbox", "{snap}");
}

#[tokio::test]
async fn semantic_snapshot_keeps_visible_content_after_hidden_node_pressure() {
    let f = fixture_with(|st| st.semantic_large_page = true).await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot(&f, &target, &tab).await;

    assert_eq!(snap["status"], "ok", "{snap}");
    assert_eq!(snap["snapshot"]["format"], "semantic_v2", "{snap}");
    assert!(
        snap["outline"]
            .as_str()
            .is_some_and(|outline| outline.contains("Visible message")
                && outline.contains("Please review the attached fixture report.")),
        "visible semantic content was omitted: {snap}"
    );
    let refs = snap["refs"].as_array().expect("semantic refs");
    assert!(
        refs.iter().any(|entry| entry["name"] == "Reply"
            && entry["actions"]
                .as_array()
                .is_some_and(|actions| actions.iter().any(|action| action == "click"))),
        "visible Reply action was omitted: {snap}"
    );
    assert!(
        refs.iter().any(|entry| entry["name"] == "Reply body"
            && entry["actions"]
                .as_array()
                .is_some_and(|actions| actions.iter().any(|action| action == "type"))),
        "visible editor was omitted: {snap}"
    );
    assert!(
        refs.iter().all(|entry| !entry["name"]
            .as_str()
            .unwrap_or("")
            .starts_with("Retained control")),
        "CSS-hidden retained controls leaked into refs: {snap}"
    );
    assert_eq!(snap["snapshot"]["omitted"]["css_hidden"], 320);
}

#[tokio::test]
async fn semantic_snapshot_can_capture_an_inactive_tab_without_activation_calls() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let result = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "session": SESSION,
            "snapshot_format": "semantic_v2",
            "include_screenshot": true
        }))
        .await;
    let snapshot = structured(&result);
    assert_eq!(snapshot["status"], "ok", "{snapshot}");
    assert_eq!(snapshot["screenshot"]["mime_type"], "image/png");
    assert_eq!(snapshot["screenshot"]["width"], 1);
    assert_eq!(snapshot["screenshot"]["height"], 1);
    assert_eq!(snapshot["screenshot"]["source"], "cdp_tab");
    assert_eq!(snapshot["screenshot_width"], 1);
    assert_eq!(snapshot["screenshot_height"], 1);
    assert_eq!(snapshot["screenshot_mime_type"], "image/png");
    assert_eq!(
        snapshot["screenshot"]["coordinate_space"],
        "viewport_css_px"
    );
    assert_eq!(snapshot["screenshot"]["viewport_css_width"], 800.0);
    assert_eq!(snapshot["screenshot"]["viewport_css_height"], 600.0);
    assert_eq!(snapshot["screenshot"]["pixel_to_css_scale_x"], 800.0);
    assert_eq!(snapshot["screenshot"]["pixel_to_css_scale_y"], 600.0);
    assert!(result.content.iter().any(|content| matches!(
        content,
        Content::Image { mime_type, .. } if mime_type == "image/png"
    )));

    let state = f.state.lock().unwrap();
    assert!(state.calls.iter().any(|(_, method, params)| {
        method == "Page.captureScreenshot"
            && params["format"] == "png"
            && params["fromSurface"] == true
            && params["captureBeyondViewport"] == false
            && params["clip"]["x"] == 0.0
            && params["clip"]["y"] == 0.0
            && params["clip"]["width"] == 800.0
            && params["clip"]["height"] == 600.0
            && params["clip"]["scale"] == 1.0
    }));
    assert!(state.calls.iter().all(|(_, method, _)| {
        method != "Target.activateTarget" && method != "Page.bringToFront"
    }));
}

#[tokio::test]
async fn semantic_snapshot_does_not_capture_unless_requested() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snapshot = semantic_snapshot(&f, &target, &tab).await;
    assert_eq!(snapshot["status"], "ok", "{snapshot}");
    assert_eq!(snapshot["screenshot"], Value::Null);
    assert_eq!(snapshot["screenshot_width"], Value::Null);
    assert_eq!(snapshot["screenshot_height"], Value::Null);
    assert_eq!(snapshot["screenshot_mime_type"], Value::Null);
    assert!(recorded_calls(&f, "Page.captureScreenshot").is_empty());
}

#[tokio::test]
async fn requested_tab_screenshot_maps_non_unit_png_pixels_to_viewport_css() {
    let f = fixture_with(|state| {
        state.viewport_css_width = 2.0;
        state.viewport_css_height = 1.0;
        state.screenshot_data = screenshot_png_base64(4, 2);
    })
    .await;
    let (target, tab) = bind(&f).await;
    let result = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "session": SESSION,
            "snapshot_format": "dom_refs_v1",
            "include_screenshot": true
        }))
        .await;
    let snapshot = structured(&result);
    assert_eq!(snapshot["status"], "ok", "{snapshot}");
    assert_eq!(snapshot["screenshot_width"], 4);
    assert_eq!(snapshot["screenshot_height"], 2);
    assert_eq!(snapshot["screenshot_mime_type"], "image/png");
    assert_eq!(snapshot["screenshot"]["viewport_css_width"], 2.0);
    assert_eq!(snapshot["screenshot"]["viewport_css_height"], 1.0);
    assert_eq!(snapshot["screenshot"]["pixel_to_css_scale_x"], 0.5);
    assert_eq!(snapshot["screenshot"]["pixel_to_css_scale_y"], 0.5);
    let capture = recorded_calls(&f, "Page.captureScreenshot");
    assert_eq!(capture.len(), 1, "{capture:?}");
    assert_eq!(capture[0].1["clip"]["width"], 2.0);
    assert_eq!(capture[0].1["clip"]["height"], 1.0);
    assert_eq!(capture[0].1["clip"]["scale"], 1.0);
}

#[tokio::test]
async fn requested_tab_screenshot_refuses_invalid_viewport_metrics_before_capture() {
    let f = fixture_with(|state| state.viewport_css_width = 0.0).await;
    let (target, tab) = bind(&f).await;
    let result = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "session": SESSION,
            "snapshot_format": "dom_refs_v1",
            "include_screenshot": true
        }))
        .await;
    let refusal = structured(&result);
    assert_eq!(refusal["status"], "refused", "{refusal}");
    assert_eq!(refusal["refusal"]["code"], "browser_route_unavailable");
    assert!(recorded_calls(&f, "Page.captureScreenshot").is_empty());
}

#[tokio::test]
async fn requested_tab_screenshot_refuses_malformed_image_data() {
    let f = fixture_with(|state| state.screenshot_data = "not-base64".into()).await;
    let (target, tab) = bind(&f).await;
    let result = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "session": SESSION,
            "snapshot_format": "semantic_v2",
            "include_screenshot": true
        }))
        .await;
    let refusal = structured(&result);
    assert_eq!(refusal["status"], "refused", "{refusal}");
    assert_eq!(refusal["refusal"]["code"], "browser_route_unavailable");
    assert!(result
        .content
        .iter()
        .all(|content| !matches!(content, Content::Image { .. })));
}

#[tokio::test]
async fn semantic_continuation_is_opaque_single_use_and_reaches_offscreen_content() {
    let f = fixture_with(|st| st.semantic_large_page = true).await;
    let (target, tab) = bind(&f).await;
    let first = semantic_snapshot(&f, &target, &tab).await;
    let token = first["snapshot"]["continuation"]
        .as_str()
        .expect("large fixture continuation")
        .to_owned();
    assert!(token.starts_with("bc-"), "{first}");

    let continued = semantic_snapshot_with(&f, &target, &tab, json!({"continuation": token})).await;
    assert_eq!(continued["status"], "ok", "{continued}");
    assert_eq!(continued["snapshot"]["scope"], "continuation");
    assert_eq!(continued["page"]["title"], "Fixture inbox", "{continued}");
    assert!(
        continued["refs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| { entry["name"] == "Archive item 304" }),
        "last offscreen action was not reachable: {continued}"
    );

    let reused = semantic_snapshot_with(&f, &target, &tab, json!({"continuation": token})).await;
    assert_eq!(reused["status"], "refused", "{reused}");
    assert_eq!(reused["refusal"]["code"], "browser_ref_stale");
}

#[tokio::test]
async fn newer_semantic_snapshot_invalidates_prior_continuations() {
    let f = fixture_with(|st| st.semantic_large_page = true).await;
    let (target, tab) = bind(&f).await;
    let first = semantic_snapshot(&f, &target, &tab).await;
    let token = first["snapshot"]["continuation"]
        .as_str()
        .expect("large fixture continuation")
        .to_owned();
    let newer = semantic_snapshot(&f, &target, &tab).await;
    assert_eq!(newer["status"], "ok", "{newer}");

    let stale = semantic_snapshot_with(&f, &target, &tab, json!({"continuation": token})).await;
    assert_eq!(stale["status"], "refused", "{stale}");
    assert_eq!(stale["refusal"]["code"], "browser_ref_stale");
}

#[tokio::test]
async fn main_frame_navigation_invalidates_semantic_continuations() {
    let f = fixture_with(|st| st.semantic_large_page = true).await;
    let (target, tab) = bind(&f).await;
    let first = semantic_snapshot(&f, &target, &tab).await;
    let token = first["snapshot"]["continuation"]
        .as_str()
        .expect("large fixture continuation")
        .to_owned();

    f.state.lock().unwrap().main_loader = "L_MAIN_2".into();

    let stale = semantic_snapshot_with(&f, &target, &tab, json!({"continuation": token})).await;
    assert_eq!(stale["status"], "refused", "{stale}");
    assert_eq!(stale["refusal"]["code"], "browser_ref_stale");
}

#[tokio::test]
async fn semantic_query_and_content_scope_are_read_only_and_precise() {
    let f = fixture_with(|st| st.semantic_large_page = true).await;
    let (target, tab) = bind(&f).await;
    let queried =
        semantic_snapshot_with(&f, &target, &tab, json!({"query": "Archive item 304"})).await;
    assert_eq!(queried["snapshot"]["scope"], "query", "{queried}");
    assert_eq!(queried["refs"].as_array().unwrap().len(), 1, "{queried}");
    assert_eq!(queried["refs"][0]["name"], "Archive item 304");

    let natural =
        semantic_snapshot_with(&f, &target, &tab, json!({"query": "reply archive 304"})).await;
    assert_eq!(natural["snapshot"]["scope"], "query", "{natural}");
    assert_eq!(natural["refs"][0]["name"], "Archive item 304", "{natural}");
    assert!(
        natural["refs"]
            .as_array()
            .is_some_and(|refs| refs.iter().any(|entry| entry["name"] == "Reply")),
        "{natural}"
    );

    let fresh = semantic_snapshot(&f, &target, &tab).await;
    let heading_ref = fresh["content_refs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "Visible message")
        .and_then(|entry| entry["ref"].as_str())
        .expect("heading content ref")
        .to_owned();
    let scoped = semantic_snapshot_with(&f, &target, &tab, json!({"scope_ref": heading_ref})).await;
    assert_eq!(scoped["snapshot"]["scope"], "subtree", "{scoped}");
    assert_eq!(scoped["refs"].as_array().unwrap().len(), 0, "{scoped}");
    assert_eq!(
        scoped["content_refs"].as_array().unwrap().len(),
        1,
        "{scoped}"
    );
    assert_eq!(scoped["content_refs"][0]["name"], "Visible message");
    assert!(recorded_calls(&f, "Page.bringToFront").is_empty());
    assert!(recorded_calls(&f, "Target.activateTarget").is_empty());
}

#[tokio::test]
async fn navigation_targets_an_inactive_tab_without_activating_it() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let result = BrowserNavigateTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "url": "https://fixture.test/background-navigation",
            "session": SESSION,
        }))
        .await;

    let structured = result
        .structured_content
        .as_ref()
        .expect("navigation structured content");
    assert_eq!(structured["status"], "ok", "{structured}");
    let navigate = recorded_calls(&f, "Page.navigate");
    assert_eq!(navigate.len(), 1, "{navigate:?}");
    assert_eq!(
        navigate[0].1["url"],
        "https://fixture.test/background-navigation"
    );
    assert!(recorded_calls(&f, "Page.bringToFront").is_empty());
    assert!(recorded_calls(&f, "Target.activateTarget").is_empty());
}

#[tokio::test]
async fn browser_visual_feedback_probes_live_tab_visibility_without_activating_it() {
    let f = fixture_with(|state| state.tab_visible = false).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let main_ref = ref_of(&snap, "main", "main-btn");

    let result = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "ref": main_ref,
            "input_route": "dom_event",
            "session": SESSION,
        }))
        .await;

    assert_eq!(
        structured(&result)["status"],
        "ok",
        "{}",
        structured(&result)
    );
    let visibility = recorded_calls(&f, "Runtime.evaluate");
    assert_eq!(visibility.len(), 1, "{visibility:?}");
    assert_eq!(
        visibility[0].1["expression"],
        "document.visibilityState === 'visible'"
    );
    assert_eq!(visibility[0].1["returnByValue"], true);
    assert_eq!(visibility[0].1["awaitPromise"], false);
    assert!(recorded_calls(&f, "Page.bringToFront").is_empty());
    assert!(recorded_calls(&f, "Target.activateTarget").is_empty());
}

#[tokio::test]
async fn semantic_refs_enforce_declared_action_kinds_before_delivery() {
    let f = fixture_with(|st| st.semantic_large_page = true).await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot(&f, &target, &tab).await;
    let content_ref = snap["content_refs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "Visible message")
        .and_then(|entry| entry["ref"].as_str())
        .unwrap();
    let click = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "session": SESSION,
            "ref": content_ref,
            "input_route": "dom_event"
        }))
        .await;
    assert_eq!(
        structured(&click)["refusal"]["code"],
        "browser_action_unavailable"
    );

    let pointer = BrowserPointerTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "session": SESSION,
            "ref": content_ref,
            "action": "hover",
            "input_route": "dom_event"
        }))
        .await;
    assert_eq!(
        structured(&pointer)["refusal"]["code"],
        "browser_action_unavailable"
    );

    let button_ref = snap["refs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "Reply")
        .and_then(|entry| entry["ref"].as_str())
        .unwrap();
    let typed = BrowserTypeTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "session": SESSION,
            "ref": button_ref,
            "text": "must not deliver"
        }))
        .await;
    assert_eq!(
        structured(&typed)["refusal"]["code"],
        "browser_action_unavailable"
    );
    assert!(recorded_calls(&f, "Input.insertText").is_empty());
    assert!(recorded_calls(&f, "Runtime.callFunctionOn").is_empty());
}

#[tokio::test]
async fn semantic_snapshot_uses_bounded_dom_fallback_only_for_known_size_failures() {
    let f = fixture_with(|st| {
        st.semantic_large_page = true;
        st.semantic_full_dom_fails = true;
    })
    .await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot_with(&f, &target, &tab, json!({"query": "Reply"})).await;
    assert_eq!(snap["status"], "ok", "{snap}");
    assert_eq!(snap["snapshot"]["complete"], false, "{snap}");
    let document_calls = recorded_calls(&f, "DOM.getDocument");
    assert!(document_calls
        .iter()
        .any(|(_, params)| params["depth"] == -1));
    assert!(document_calls
        .iter()
        .any(|(_, params)| params["depth"] == 256));
    assert!(document_calls
        .iter()
        .any(|(_, params)| params["depth"] == 8));
}

#[tokio::test]
async fn semantic_snapshot_uses_bounded_dom_fallback_for_full_tree_timeout() {
    let f = fixture_with(|st| {
        st.semantic_large_page = true;
        st.semantic_full_dom_times_out = true;
    })
    .await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot_with(&f, &target, &tab, json!({"query": "Reply"})).await;
    assert_eq!(snap["status"], "ok", "{snap}");
    assert_eq!(snap["snapshot"]["complete"], false, "{snap}");
    let document_calls = recorded_calls(&f, "DOM.getDocument");
    assert!(document_calls
        .iter()
        .any(|(_, params)| params["depth"] == -1));
    assert!(document_calls
        .iter()
        .any(|(_, params)| params["depth"] == 8));
    assert!(document_calls.iter().all(|(_, params)| {
        params["depth"] == -1 || params["depth"].as_i64().is_some_and(|depth| depth <= 8)
    }));
}

#[tokio::test]
async fn semantic_snapshot_hydrates_truncated_fallback_branches() {
    let f = fixture_with(|st| {
        st.semantic_large_page = true;
        st.semantic_full_dom_fails = true;
        st.semantic_truncated_dom = true;
    })
    .await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot_with(&f, &target, &tab, json!({"query": "Reply"})).await;
    assert_eq!(snap["status"], "ok", "{snap}");
    assert_eq!(snap["snapshot"]["complete"], false, "{snap}");
    assert!(snap["refs"]
        .as_array()
        .is_some_and(|refs| refs.iter().any(|entry| entry["name"] == "Reply")));
    assert!(recorded_calls(&f, "DOM.describeNode")
        .iter()
        .any(|(_, params)| params["backendNodeId"] == 999));
}

#[tokio::test]
async fn unproven_oopif_capability_is_omitted_not_guessed() {
    let f = fixture_with(|st| st.oopif_supported = false).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;

    assert_eq!(
        snap["status"], "ok",
        "capability gap is not an error: {snap}"
    );
    assert_eq!(snap["oopif"]["status"], "unsupported");
    assert_eq!(snap["oopif"]["frames"], 0);
    let frames: Vec<&str> = snap["refs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["frame"].as_str().unwrap())
        .collect();
    assert_eq!(
        frames,
        vec!["main", "main", "main", "iframe"],
        "OOPIF content omitted, everything provable kept: {snap}"
    );
}

#[tokio::test]
async fn rogue_attach_announcements_are_contained() {
    let f = fixture_with(|st| st.emit_rogue_attach = true).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;

    let oopif_refs: Vec<&Value> = snap["refs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["frame"] == "oopif")
        .collect();
    assert_eq!(
        oopif_refs.len(),
        1,
        "only the child attached beneath the proven tab session mints refs: {snap}"
    );
    assert_eq!(snap["oopif"]["frames"], 1);

    // The rogue/popup sessions must never have been spoken to.
    let touched: Vec<String> = f
        .state
        .lock()
        .unwrap()
        .calls
        .iter()
        .filter_map(|(s, _, _)| s.clone())
        .collect();
    assert!(
        !touched
            .iter()
            .any(|s| s == "rogue-sess" || s == "popup-sess"),
        "core issued commands to an unproven session: {touched:?}"
    );
}

// ── Mutation routing + frame identity revalidation ──────────────────────────

#[tokio::test]
async fn protected_browser_scope_reproves_live_origin_and_omits_sensitive_url_text() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let args = json!({
        "target_id": target,
        "tab_id": tab,
        "session": SESSION,
        "x": 10,
        "y": 20,
    });

    let first = browser_protected_resource_scope(&f.engine, &args, "browser_click")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first["live_origin"], "https://fixture.test");
    assert!(
        !first.to_string().contains("secret"),
        "the resource must not contain a full URL"
    );
    let observation = browser_protected_resource_scope(&f.engine, &args, "get_browser_state")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(observation["action_class"], "page_observation");
    assert_eq!(first["action_class"], "page_input");

    f.state.lock().unwrap().main_url = "https://bank.example/transfer?secret=one-time-token".into();
    let second = browser_protected_resource_scope(&f.engine, &args, "browser_click")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(second["live_origin"], "https://bank.example");
    assert_ne!(
        first, second,
        "cross-origin navigation must rotate the grant scope"
    );
    assert!(
        !second.to_string().contains("one-time-token"),
        "query text must never reach the consent resource"
    );
}

#[tokio::test]
async fn click_routes_oopif_refs_through_the_contained_child_session() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let oopif_ref = ref_of(&snap, "oopif", "ad-input");

    let result = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": oopif_ref, "session": SESSION
        }))
        .await;
    let s = structured(&result);
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["frame"], "oopif");
    // Box-model center of backend 100 in the child session's space.
    assert_eq!(s["x"], 1010.0);
    assert_eq!(s["y"], 1005.0);

    let mouse = recorded_calls(&f, "Input.dispatchMouseEvent");
    assert!(!mouse.is_empty());
    assert!(
        mouse
            .iter()
            .all(|(sess, _)| sess.as_deref().unwrap_or("").starts_with("oopif-sess-")),
        "OOPIF input must dispatch on the child session: {mouse:?}"
    );
    let focus_emulation = recorded_calls(&f, "Emulation.setFocusEmulationEnabled");
    assert_eq!(focus_emulation.len(), 2);
    assert_eq!(focus_emulation[0].1["enabled"], true);
    assert_eq!(focus_emulation[1].1["enabled"], false);
    assert!(recorded_calls(&f, "Page.bringToFront").is_empty());
    assert!(recorded_calls(&f, "Target.activateTarget").is_empty());
}

#[tokio::test]
async fn click_validates_same_process_iframe_loader_and_uses_the_tab_session() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let iframe_ref = ref_of(&snap, "iframe", "inner-btn");

    let result = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": iframe_ref, "session": SESSION
        }))
        .await;
    let s = structured(&result);
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["frame"], "iframe");

    let mouse = recorded_calls(&f, "Input.dispatchMouseEvent");
    assert!(
        mouse
            .iter()
            .all(|(sess, _)| sess.as_deref().unwrap_or("").starts_with("tab-sess-")),
        "same-process iframe input stays on the tab session: {mouse:?}"
    );
    assert!(
        !recorded_calls(&f, "Page.getFrameTree").is_empty(),
        "frame identity must have been re-proven"
    );
}

#[tokio::test]
async fn semantic_snapshot_reports_the_live_page_title() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot(&f, &target, &tab).await;
    assert_eq!(snap["status"], "ok", "{snap}");
    assert_eq!(snap["page"]["title"], "Fixture after navigation", "{snap}");
}

#[tokio::test]
async fn foreground_delivery_mode_accepts_window_activation_for_trusted_click() {
    let f = fixture_with_platform(|_| {}, true).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let main_ref = ref_of(&snap, "main", "main-btn");

    let bad = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": main_ref,
            "delivery_mode": "sideways", "session": SESSION
        }))
        .await;
    assert!(
        bad.is_error.unwrap_or(false),
        "an unknown delivery_mode is refused"
    );
    assert!(recorded_calls(&f, "Input.dispatchMouseEvent").is_empty());

    let refused = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": main_ref, "session": SESSION
        }))
        .await;
    assert_eq!(
        structured(&refused)["refusal"]["detail"]["alternative_delivery_mode"],
        "foreground"
    );

    let foreground = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": main_ref,
            "delivery_mode": "foreground", "session": SESSION
        }))
        .await;
    let s = structured(&foreground);
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["route"], "trusted");
    assert_eq!(s["delivery_mode"], "foreground");
    let mouse = recorded_calls(&f, "Input.dispatchMouseEvent");
    assert_eq!(mouse.len(), 2, "press and release: {mouse:?}");
}

#[tokio::test]
async fn trusted_click_refuses_when_standalone_background_posture_is_unavailable() {
    let f = fixture_with_platform(|_| {}, true).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let main_ref = ref_of(&snap, "main", "main-btn");

    let trusted = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": main_ref,
            "input_route": "trusted", "session": SESSION
        }))
        .await;
    assert_eq!(
        structured(&trusted)["refusal"]["code"],
        "browser_input_trust_unavailable"
    );
    assert_eq!(
        structured(&trusted)["refusal"]["detail"]["alternative_route"],
        "dom_event"
    );
    assert_eq!(
        structured(&trusted)["refusal"]["detail"]["trusted_delivery_attempted"],
        false
    );
    assert!(recorded_calls(&f, "Input.dispatchMouseEvent").is_empty());

    let synthetic_args = json!({
        "target_id": target, "tab_id": tab, "ref": main_ref,
        "input_route": "dom_event", "session": SESSION
    });
    let synthetic = BrowserClickTool::new(f.engine.clone())
        .invoke(synthetic_args.clone())
        .await;
    assert_eq!(structured(&synthetic)["status"], "ok");
    assert_eq!(structured(&synthetic)["effect"], "unverifiable");
    assert_eq!(structured(&synthetic)["escalation"]["recommended"], "page");
    assert!(synthetic.content.iter().any(|content| matches!(
        content,
        crate::protocol::Content::Text { text, .. }
            if text.contains("application effect not verified")
                && text.contains("trust-gated controls")
    )));
    let public = ActionExecutionRecord::from_legacy(
        "browser_click",
        &synthetic_args,
        structured(&synthetic),
    )
    .expect("browser click action record")
    .public_result()
    .expect("public browser click result");
    let public = serde_json::to_value(public).expect("serialize public result");
    assert_eq!(public["effect"], "unverifiable", "{public}");
    assert_eq!(public["route"], "dom", "{public}");
    assert_eq!(public["delivery"]["mode"], "background", "{public}");
    assert_eq!(public["escalation"]["target"], "page", "{public}");
    assert_eq!(
        public["escalation"]["reason"], "effect_unconfirmed",
        "{public}"
    );
    assert!(public.get("status").is_none(), "{public}");
    assert!(!recorded_calls(&f, "Runtime.callFunctionOn").is_empty());
    assert!(recorded_calls(&f, "Page.bringToFront").is_empty());
    assert!(recorded_calls(&f, "Target.activateTarget").is_empty());
}

#[tokio::test]
async fn main_frame_navigation_invalidates_refs_via_loader_identity() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let main_ref = ref_of(&snap, "main", "main-btn");

    f.state.lock().unwrap().main_loader = "L_MAIN_2".into();

    let click = BrowserClickTool::new(f.engine.clone());
    let result = click
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": main_ref, "session": SESSION
        }))
        .await;
    assert_eq!(
        structured(&result)["refusal"]["code"],
        "browser_ref_stale",
        "loader change means a new document"
    );
    assert!(
        recorded_calls(&f, "Input.dispatchMouseEvent").is_empty(),
        "no input may reach a document that cannot be re-proven"
    );

    // The whole snapshot namespace was invalidated, so a retry refuses
    // at resolution already.
    let retry = click
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": main_ref, "session": SESSION
        }))
        .await;
    assert_eq!(structured(&retry)["refusal"]["code"], "browser_ref_stale");
}

#[tokio::test]
async fn same_process_iframe_navigation_invalidates_its_refs() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let iframe_ref = ref_of(&snap, "iframe", "inner-btn");

    f.state.lock().unwrap().iframe_loader = "L_IFRAME_2".into();

    let result = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": iframe_ref, "session": SESSION
        }))
        .await;
    assert_eq!(structured(&result)["refusal"]["code"], "browser_ref_stale");
    assert!(recorded_calls(&f, "Input.dispatchMouseEvent").is_empty());
}

#[tokio::test]
async fn oopif_navigation_invalidates_its_refs() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let oopif_ref = ref_of(&snap, "oopif", "ad-input");

    f.state.lock().unwrap().oopif_loader = "L_OOPIF_2".into();

    let result = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": oopif_ref, "session": SESSION
        }))
        .await;
    assert_eq!(structured(&result)["refusal"]["code"], "browser_ref_stale");
    assert!(recorded_calls(&f, "Input.dispatchMouseEvent").is_empty());
}

#[tokio::test]
async fn removed_oopif_frame_is_stale_not_guessed() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let oopif_ref = ref_of(&snap, "oopif", "ad-input");

    f.state.lock().unwrap().oopif_present = false;

    let result = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": oopif_ref, "session": SESSION
        }))
        .await;
    assert_eq!(structured(&result)["refusal"]["code"], "browser_ref_stale");
    assert!(recorded_calls(&f, "Input.dispatchMouseEvent").is_empty());
}

#[tokio::test]
async fn oopif_capability_regression_refuses_rather_than_reroutes() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let oopif_ref = ref_of(&snap, "oopif", "ad-input");

    f.state.lock().unwrap().oopif_supported = false;

    let result = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": oopif_ref, "session": SESSION
        }))
        .await;
    assert_eq!(
        structured(&result)["refusal"]["code"],
        "browser_route_unavailable",
        "a lost capability is a refusal, never a fallback to the wrong session"
    );
    assert!(recorded_calls(&f, "Input.dispatchMouseEvent").is_empty());
}

// ── Typing routes ────────────────────────────────────────────────────────────

#[tokio::test]
async fn typing_into_composed_shadow_input_uses_the_tab_session() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let shadow_ref = ref_of(&snap, "main", "Shadow Input");

    let result = BrowserTypeTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": shadow_ref,
            "text": "hi", "session": SESSION
        }))
        .await;
    let s = structured(&result);
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["frame"], "main");

    let inserts = recorded_calls(&f, "Input.insertText");
    assert_eq!(inserts.len(), 1);
    assert!(inserts[0].0.as_deref().unwrap().starts_with("tab-sess-"));
    assert_eq!(inserts[0].1["text"], "hi");
    assert!(
        !recorded_calls(&f, "Runtime.callFunctionOn").is_empty(),
        "editability must be checked on the node itself"
    );
    assert!(recorded_calls(&f, "Page.bringToFront").is_empty());
    assert!(recorded_calls(&f, "Target.activateTarget").is_empty());
}

#[tokio::test]
async fn typing_into_an_oopif_input_routes_to_the_child_session() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let oopif_ref = ref_of(&snap, "oopif", "ad-input");

    let result = BrowserTypeTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": oopif_ref,
            "text": "q", "session": SESSION
        }))
        .await;
    let s = structured(&result);
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["frame"], "oopif");

    let inserts = recorded_calls(&f, "Input.insertText");
    assert_eq!(inserts.len(), 1);
    assert!(
        inserts[0].0.as_deref().unwrap().starts_with("oopif-sess-"),
        "typing must land in the contained child session: {inserts:?}"
    );
    let focuses = recorded_calls(&f, "DOM.focus");
    assert!(focuses
        .iter()
        .all(|(sess, _)| sess.as_deref().unwrap().starts_with("oopif-sess-")));
}

#[tokio::test]
async fn partial_keystrokes_report_exact_delivered_prefix() {
    let f = fixture_with(|state| state.fail_key_down_after = Some(2)).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let shadow_ref = ref_of(&snap, "main", "Shadow Input");

    let result = BrowserTypeTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "ref": shadow_ref,
            "text": "four",
            "mode": "keystrokes",
            "session": SESSION
        }))
        .await;
    let refusal = &structured(&result)["refusal"];
    assert_eq!(refusal["code"], "browser_input_incomplete");
    assert_eq!(refusal["detail"]["requested_chars"], 4);
    assert_eq!(refusal["detail"]["delivered_chars"], 2);
    assert_eq!(refusal["detail"]["retryable"], false);
}

#[tokio::test]
async fn keystrokes_use_char_events_for_text_delivery() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let shadow_ref = ref_of(&snap, "main", "Shadow Input");

    let result = BrowserTypeTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "ref": shadow_ref,
            "text": "a\n",
            "mode": "keystrokes",
            "session": SESSION
        }))
        .await;
    assert_eq!(structured(&result)["status"], "ok");

    let events = recorded_calls(&f, "Input.dispatchKeyEvent");
    assert_eq!(events.len(), 6);
    let params: Vec<&Value> = events.iter().map(|(_, params)| params).collect();
    assert_eq!(params[0]["type"], "keyDown");
    assert!(params[0].get("text").is_none());
    assert_eq!(params[1]["type"], "char");
    assert_eq!(params[1]["text"], "a");
    assert_eq!(params[1]["unmodifiedText"], "a");
    assert_eq!(params[2]["type"], "keyUp");
    assert!(params[2].get("text").is_none());
    assert_eq!(params[3]["type"], "keyDown");
    assert_eq!(params[3]["key"], "Enter");
    assert_eq!(params[4]["type"], "char");
    assert_eq!(params[4]["key"], "Enter");
    assert_eq!(params[4]["text"], "\r");
    assert_eq!(params[4]["unmodifiedText"], "\r");
    assert_eq!(params[5]["type"], "keyUp");
    let focus_emulation = recorded_calls(&f, "Emulation.setFocusEmulationEnabled");
    assert_eq!(focus_emulation.len(), 2);
    assert_eq!(focus_emulation[0].1["enabled"], true);
    assert_eq!(focus_emulation[1].1["enabled"], false);
    let readiness_checks = recorded_calls(&f, "Runtime.callFunctionOn");
    assert!(readiness_checks.iter().any(|(_, params)| {
        params["functionDeclaration"]
            .as_str()
            .is_some_and(|declaration| {
                declaration.contains("document.hasFocus()")
                    && declaration.contains("active === this")
            })
    }));
    assert!(recorded_calls(&f, "Page.bringToFront").is_empty());
    assert!(recorded_calls(&f, "Target.activateTarget").is_empty());
}

// ---- window-less page targets (trycua/cua#3540) ------------------------------------------
//
// Chromium exposes page targets that live in no browser window — an extension
// side panel, an offscreen document — and answers `Browser.getWindowForTarget`
// for them with `-32000 Browser window not found`. Such a target can never be
// the tab of the requested native window, so candidate enumeration skips it;
// every other error keeps failing the whole proof.

fn side_panel_target() -> Value {
    json!({
        "targetId": "T_SIDEPANEL",
        "type": "page",
        "title": "Claude",
        "url": "chrome-extension://fcoeoabgfenejglbffodgkkbkcdhcgfn/sidepanel.html?tabId=42",
        "attached": false,
    })
}

async fn bind_result(f: &Fixture) -> Value {
    let tool = GetBrowserStateTool::new(f.engine.clone());
    let result = tool
        .invoke(json!({ "pid": 1, "window_id": 7, "session": SESSION }))
        .await;
    structured(&result).clone()
}

#[tokio::test]
async fn windowless_side_panel_target_is_skipped_and_bind_stays_exact() {
    let f = fixture_with(|st| st.windowless_targets.push(side_panel_target())).await;
    let s = bind_result(&f).await;
    assert_eq!(s["status"], "ok", "bind must succeed: {s}");
    assert_eq!(s["binding_quality"], "exact");
    let tabs = s["tabs"].as_array().expect("tabs");
    assert_eq!(
        tabs.len(),
        1,
        "the side panel is not a tab of the bound window: {s}"
    );
    assert_eq!(tabs[0]["url"], "https://fixture.test/");
    // The window-less target was asked for its window (and skipped), never attached to.
    let st = f.state.lock().unwrap();
    assert!(st
        .calls
        .iter()
        .any(|(_, m, p)| { m == "Browser.getWindowForTarget" && p["targetId"] == "T_SIDEPANEL" }));
    assert!(!st
        .calls
        .iter()
        .any(|(_, m, p)| { m == "Target.attachToTarget" && p["targetId"] == "T_SIDEPANEL" }));
}

#[tokio::test]
async fn only_windowless_targets_refuse_wrong_target_never_route_unavailable() {
    let f = fixture_with(|st| {
        st.omit_primary_target = true;
        st.windowless_targets.push(side_panel_target());
    })
    .await;
    let s = bind_result(&f).await;
    assert_eq!(s["status"], "refused", "{s}");
    assert_eq!(
        s["refusal"]["code"], "browser_wrong_target_refused",
        "no windowed candidate → refuse rather than guess, and never a route failure: {s}"
    );
}

#[tokio::test]
async fn other_window_lookup_errors_still_fail_the_whole_proof() {
    let f = fixture_with(|st| {
        st.primary_window_error = Some((-32000, "Target closed".into()));
    })
    .await;
    let s = bind_result(&f).await;
    assert_eq!(s["status"], "refused", "{s}");
    assert_eq!(s["refusal"]["code"], "browser_route_unavailable", "{s}");
    assert!(
        s["refusal"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("Browser.getWindowForTarget failed while proving the native window"),
        "{s}"
    );
}

#[tokio::test]
async fn method_unsupported_keeps_the_electron_none_path() {
    // Electron omits the Browser domain method entirely: the candidate keeps a
    // None geometry and binds through the single-page, single-native-window
    // cardinality proof; the window-less skip must not drop it.
    let f = fixture_with(|st| {
        st.primary_window_error =
            Some((-32601, "'Browser.getWindowForTarget' wasn't found".into()));
    })
    .await;
    let s = bind_result(&f).await;
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["binding_quality"], "exact", "{s}");
    let tabs = s["tabs"].as_array().expect("tabs");
    assert_eq!(tabs.len(), 1, "{s}");
    assert_eq!(tabs[0]["url"], "https://fixture.test/");
}

// ── File choosers, uploads, and drops ────────────────────────────────────────

fn upload_fixture_files(count: usize) -> (tempfile::TempDir, Vec<String>) {
    let directory = tempfile::tempdir().unwrap();
    let files = (0..count)
        .map(|index| {
            let path = directory.path().join(format!("upload-{index}.txt"));
            std::fs::write(&path, b"fixture").unwrap();
            std::fs::canonicalize(path)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    (directory, files)
}

async fn upload(f: &Fixture, target: &str, tab: &str, reference: &str, files: &[String]) -> Value {
    let result = BrowserSetInputFilesTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": reference,
            "files": files, "session": SESSION
        }))
        .await;
    structured(&result).clone()
}

/// After a tool call no session may still intercept: the person's own
/// clicks must get the native picker back.
fn assert_interception_released(f: &Fixture) {
    let st = f.state.lock().unwrap();
    assert!(
        st.intercepting.is_empty(),
        "interception left on: {:?}",
        st.intercepting
    );
    assert_eq!(
        st.native_pickers_shown, 0,
        "a native picker reached the desktop"
    );
}

#[tokio::test]
async fn trusted_click_contains_the_file_chooser_it_triggers_and_says_so() {
    let f = fixture_with(|st| st.chooser_on_activation = Some((Some(41), "selectSingle"))).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");

    let result = BrowserClickTool::new(f.engine.clone())
        .invoke(json!({ "target_id": target, "tab_id": tab, "ref": button, "session": SESSION }))
        .await;
    let s = structured(&result);
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["file_chooser"]["requested"], true, "{s}");
    assert_eq!(s["file_chooser"]["shown"], false);
    assert_eq!(s["file_chooser"]["next_action"], "browser_set_input_files");
    let Content::Text { text, .. } = &result.content[0] else {
        panic!("text result");
    };
    assert!(
        text.contains(&format!("browser_set_input_files with ref {button}")),
        "{text}"
    );
    assert_interception_released(&f);

    // Interception ran on its own attachment with Page enabled and was
    // cancelled for an action that only clicks.
    let intercept = recorded_calls(&f, "Page.setInterceptFileChooserDialog");
    assert_eq!(intercept.first().unwrap().1["cancel"], true);
    let interception_session = intercept.first().unwrap().0.clone().unwrap();
    assert!(recorded_calls(&f, "Page.enable")
        .iter()
        .any(|(session, _)| session.as_deref() == Some(interception_session.as_str())));
    assert!(recorded_calls(&f, "Target.detachFromTarget")
        .iter()
        .any(|(_, params)| params["sessionId"] == interception_session.as_str()));
}

#[tokio::test]
async fn interception_covers_out_of_process_iframes() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    BrowserClickTool::new(f.engine.clone())
        .invoke(json!({ "target_id": target, "tab_id": tab, "ref": button, "session": SESSION }))
        .await;
    let enabled_on: Vec<String> = recorded_calls(&f, "Page.setInterceptFileChooserDialog")
        .into_iter()
        .filter(|(_, params)| params["enabled"] == true)
        .filter_map(|(session, _)| session)
        .collect();
    assert!(
        enabled_on.iter().any(|s| s.starts_with("tab-sess-")),
        "{enabled_on:?}"
    );
    assert!(
        enabled_on.iter().any(|s| s.starts_with("oopif-sess-")),
        "{enabled_on:?}"
    );
    let detached: Vec<Value> = recorded_calls(&f, "Target.detachFromTarget")
        .into_iter()
        .map(|(_, params)| params["sessionId"].clone())
        .collect();
    for session in &enabled_on {
        assert!(
            detached.contains(&json!(session)),
            "{session} still attached: {detached:?}"
        );
    }
    assert_interception_released(&f);
}

#[tokio::test]
async fn trusted_pointer_actions_contain_file_choosers_too() {
    let f = fixture_with(|st| st.chooser_on_activation = Some((Some(41), "selectMultiple"))).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let result = BrowserPointerTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": button,
            "action": "double_click", "session": SESSION
        }))
        .await;
    let s = structured(&result);
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["file_chooser"]["multiple"], true, "{s}");
    assert_interception_released(&f);
}

#[tokio::test]
async fn upload_fills_the_ref_itself_or_the_field_it_names_without_activation() {
    let f = fixture_with(|st| {
        st.related_field_direct = true;
        st.chooser_on_activation = Some((Some(41), "selectSingle"));
    })
    .await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(1);

    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["route"], "related_file_input");
    assert_eq!(s["native_picker_shown"], false);
    let set = recorded_calls(&f, "DOM.setFileInputFiles");
    assert_eq!(set.len(), 1);
    assert_eq!(set[0].1["backendNodeId"], 77);
    assert_eq!(set[0].1["files"], json!(files));
    assert!(
        recorded_calls(&f, "Input.dispatchMouseEvent").is_empty(),
        "no activation needed"
    );
    assert!(recorded_calls(&f, "Page.setInterceptFileChooserDialog").is_empty());
}

#[tokio::test]
async fn upload_refuses_several_files_for_a_single_file_field() {
    let f = fixture_with(|st| st.related_field_direct = true).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(2);
    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["refusal"]["code"], "browser_action_unavailable", "{s}");
    assert!(recorded_calls(&f, "DOM.setFileInputFiles").is_empty());
}

#[tokio::test]
async fn upload_fills_exactly_the_chooser_a_control_opens() {
    let f = fixture_with(|st| st.chooser_on_activation = Some((Some(41), "selectMultiple"))).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(2);

    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["route"], "file_chooser");
    assert_eq!(s["file_count"], 2);
    let set = recorded_calls(&f, "DOM.setFileInputFiles");
    assert_eq!(set.len(), 1);
    assert_eq!(set[0].1["backendNodeId"], 41, "the chooser's own element");
    let intercept = recorded_calls(&f, "Page.setInterceptFileChooserDialog");
    assert_eq!(
        set[0].0, intercept[0].0,
        "filled on the session that intercepted"
    );
    assert_eq!(
        intercept[0].1["cancel"], false,
        "a chooser about to be filled is not cancelled"
    );
    assert!(recorded_calls(&f, "Input.dispatchDragEvent").is_empty());
    assert_interception_released(&f);
}

#[tokio::test]
async fn upload_refuses_several_files_for_a_single_file_chooser() {
    let f = fixture_with(|st| st.chooser_on_activation = Some((Some(41), "selectSingle"))).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(3);
    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["refusal"]["code"], "browser_action_unavailable", "{s}");
    assert!(recorded_calls(&f, "DOM.setFileInputFiles").is_empty());
    assert_interception_released(&f);
}

#[tokio::test]
async fn upload_activates_with_a_user_gesture_where_trusted_input_would_raise_the_window() {
    let f = fixture_with_platform(
        |st| st.chooser_on_activation = Some((Some(41), "selectSingle")),
        true,
    )
    .await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(1);
    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["route"], "file_chooser", "{s}");
    assert!(recorded_calls(&f, "Input.dispatchMouseEvent").is_empty());
    assert!(recorded_calls(&f, "Runtime.callFunctionOn")
        .iter()
        .any(|(_, params)| params["userGesture"] == true));
    assert_interception_released(&f);
}

#[tokio::test]
async fn upload_uses_the_one_nearby_field_when_the_control_opens_nothing() {
    let f = fixture_with(|st| st.related_field_nearby = true).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(1);
    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["route"], "nearby_file_input", "{s}");
    assert_eq!(
        recorded_calls(&f, "DOM.setFileInputFiles")[0].1["backendNodeId"],
        77
    );
    assert_interception_released(&f);
}

#[tokio::test]
async fn upload_drops_files_on_a_zone_without_a_field_or_chooser() {
    let f = fixture().await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(2);

    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["route"], "drop");
    assert_eq!(s["effect"], "unverifiable");
    let drags = recorded_calls(&f, "Input.dispatchDragEvent");
    let kinds: Vec<&str> = drags
        .iter()
        .map(|(_, p)| p["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["dragEnter", "dragOver", "drop"]);
    for (_, params) in &drags {
        assert_eq!(params["data"]["files"], json!(files));
        assert_eq!(params["x"], 110.0, "box center of backend 10");
    }
    assert!(recorded_calls(&f, "DOM.setFileInputFiles").is_empty());
    assert_interception_released(&f);
}

#[tokio::test]
async fn upload_drops_files_when_the_page_asks_for_a_file_system_access_picker() {
    let f = fixture_with(|st| {
        st.chooser_on_activation = Some((None, "selectSingle"));
        st.related_field_nearby = true;
    })
    .await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(1);
    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(
        s["route"], "drop",
        "a File System Access page wants its own picker, not a nearby field: {s}"
    );
    assert!(recorded_calls(&f, "DOM.setFileInputFiles").is_empty());
    assert_interception_released(&f);
}

#[tokio::test]
async fn existing_profile_uploads_stay_within_the_reviewed_cdp_surface() {
    const TRANSPORT: &str = "transport-v2-upload-policy";
    let (f, _) = protected_existing_profile_fixture().await;
    f.state.lock().unwrap().chooser_on_activation = Some((Some(41), "selectMultiple"));
    let prepare = BrowserPrepareTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1, "window_id": 7, "session": SESSION,
            "_transport_session_id": TRANSPORT,
            "strategy": { "kind": "existing_profile" }
        }))
        .await;
    assert_eq!(
        structured(&prepare)["status"],
        "ok",
        "{}",
        structured(&prepare)
    );
    let state = GetBrowserStateTool::new(f.engine.clone())
        .invoke(json!({
            "pid": 1, "window_id": 7, "session": SESSION,
            "_transport_session_id": TRANSPORT
        }))
        .await;
    let bound = structured(&state).clone();
    let target = bound["target_id"].as_str().unwrap().to_owned();
    let tab = bound["tabs"][0]["tab_id"].as_str().unwrap().to_owned();
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let (_dir, files) = upload_fixture_files(2);

    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["route"], "file_chooser", "{s}");
    f.state.lock().unwrap().chooser_on_activation = None;
    let s = upload(&f, &target, &tab, &button, &files).await;
    assert_eq!(s["route"], "drop", "{s}");
    assert_interception_released(&f);

    // The policy refuses unreviewed methods before they reach the socket, so
    // every recorded call is proof the surface was enough.
    let calls = f.state.lock().unwrap().calls.clone();
    for (_, method, _) in &calls {
        assert!(
            super::cdp_ws::existing_profile_allows(method),
            "unreviewed method {method}"
        );
    }
    crate::session::fire_session_end(TRANSPORT);
}

async fn download(
    f: &Fixture,
    target: &str,
    tab: &str,
    reference: &str,
    root: &std::path::Path,
) -> Value {
    let result = super::download::BrowserDownloadTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target, "tab_id": tab, "ref": reference,
            "destination_root": root, "session": SESSION,
            super::download::MCP_HOST_DOWNLOAD_APPROVAL_ARG: true
        }))
        .await;
    structured(&result).clone()
}

#[tokio::test]
async fn download_keeps_the_site_name_and_never_overwrites() {
    let f = fixture_with(|st| {
        st.download_on_activation = Some(("../Quarterly report.pdf", b"%PDF-new"))
    })
    .await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let link = ref_of(&snap, "main", "main-btn");
    let root = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(root.path()).unwrap();
    std::fs::write(root.join("Quarterly report.pdf"), b"older").unwrap();

    let s = download(&f, &target, &tab, &link, &root).await;
    assert_eq!(s["status"], "completed", "{s}");
    assert_eq!(s["file_name"], "Quarterly report (1).pdf");
    assert_eq!(s["bytes"], 8);
    assert_eq!(
        std::fs::read(root.join("Quarterly report.pdf")).unwrap(),
        b"older"
    );
    assert_eq!(
        std::fs::read(root.join("Quarterly report (1).pdf")).unwrap(),
        b"%PDF-new"
    );
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["Quarterly report (1).pdf", "Quarterly report.pdf"]);
    let public = s.to_string();
    assert!(!public.contains("fixture.test/report"), "{public}");
    assert!(!public.contains(root.to_str().unwrap()), "{public}");
    assert_interception_released(&f);
}

#[tokio::test]
async fn download_ref_that_asks_for_a_file_chooser_is_refused_without_a_picker() {
    let f = fixture_with(|st| st.chooser_on_activation = Some((Some(41), "selectSingle"))).await;
    let (target, tab) = bind(&f).await;
    let snap = snapshot(&f, &target, &tab).await;
    let button = ref_of(&snap, "main", "main-btn");
    let root = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(root.path()).unwrap();
    let s = download(&f, &target, &tab, &button, &root).await;
    assert_eq!(s["refusal"]["code"], "browser_action_unavailable", "{s}");
    assert_eq!(
        s["refusal"]["detail"]["next_action"],
        "browser_set_input_files"
    );
    assert_interception_released(&f);
    let behaviors = recorded_calls(&f, "Browser.setDownloadBehavior");
    assert_eq!(
        behaviors.last().unwrap().1["behavior"],
        "default",
        "browser-wide routing restored"
    );
}

// ── Dropdowns ────────────────────────────────────────────────────────────────

async fn select_option(
    f: &Fixture,
    target: &str,
    tab: &str,
    reference: &str,
    option: &str,
) -> ToolResult {
    BrowserSelectTool::new(f.engine.clone())
        .invoke(json!({
            "target_id": target,
            "tab_id": tab,
            "session": SESSION,
            "ref": reference,
            "option": option,
        }))
        .await
}

fn semantic_ref_named(snapshot: &Value, name: &str) -> Value {
    snapshot["refs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("no ref named {name:?}: {snapshot}"))
        .clone()
}

fn no_input_or_activation(f: &Fixture) {
    for method in [
        "Input.dispatchMouseEvent",
        "Input.dispatchKeyEvent",
        "Input.insertText",
        "Page.bringToFront",
        "Target.activateTarget",
    ] {
        assert!(recorded_calls(f, method).is_empty(), "{method} was sent");
    }
}

#[tokio::test]
async fn native_select_lists_its_options_and_is_chosen_through_the_page() {
    let f = fixture_with(|st| {
        st.select_page = true;
        st.native_select_reply = json!({
            "kind": "native", "label": "Portugal", "value": "PT",
            "changed": true, "kept": true, "now": "Portugal"
        });
    })
    .await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot(&f, &target, &tab).await;
    let country = semantic_ref_named(&snap, "País ou região");
    assert_eq!(
        country["options"],
        json!(["Selecionar", "Países Baixos", "Portugal"])
    );
    assert!(country.get("options_total").is_none());
    assert!(country["actions"]
        .as_array()
        .unwrap()
        .contains(&json!("select")));

    let result = select_option(
        &f,
        &target,
        &tab,
        country["ref"].as_str().unwrap(),
        "portugal",
    )
    .await;
    let s = structured(&result);
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["route"], "native_select");
    assert_eq!(s["effect"], "verified");
    assert_eq!(s["label"], "Portugal");
    assert_eq!(s["native_menu_shown"], false);
    let calls = recorded_calls(&f, "Runtime.callFunctionOn");
    assert_eq!(
        calls.len(),
        1,
        "the ARIA path must not run for a native select"
    );
    assert_eq!(calls[0].1["arguments"][0]["value"], "portugal");
    assert_eq!(calls[0].1["objectId"], "obj-40");
    no_input_or_activation(&f);
}

#[tokio::test]
async fn select_without_a_match_returns_the_options_there_are() {
    let f = fixture_with(|st| {
        st.select_page = true;
        st.native_select_reply = json!({
            "kind": "native", "error": "no_match",
            "options": ["Selecionar", "Países Baixos", "Portugal"], "options_total": 3
        });
    })
    .await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot(&f, &target, &tab).await;
    let country = semantic_ref_named(&snap, "País ou região");
    let result = select_option(
        &f,
        &target,
        &tab,
        country["ref"].as_str().unwrap(),
        "Espanha",
    )
    .await;
    let s = structured(&result);
    assert_eq!(s["refusal"]["code"], "browser_action_unavailable");
    let text = match &result.content[0] {
        Content::Text { text, .. } => text.clone(),
        other => panic!("expected text, got {other:?}"),
    };
    assert!(text.contains("\"Portugal\""), "{text}");
    assert_eq!(s["refusal"]["detail"]["options_total"], 3);
}

/// Um dropdown feito à mão, como a árvore AX o mostra: fechado, aberto com
/// a lista que controla, e com a escolha feita.
fn aria_dropdown_trees() -> Vec<Value> {
    let trigger = |value: &str, expanded: bool, controls: bool| {
        let mut properties = vec![
            json!({"name": "expanded", "value": {"type": "boolean", "value": expanded}}),
            json!({"name": "hasPopup", "value": {"type": "token", "value": "listbox"}}),
        ];
        if controls {
            properties.push(
                json!({"name": "controls", "value": {"type": "idrefList", "value": "",
                "relatedNodes": [{"backendDOMNodeId": 60}]}}),
            );
        }
        json!({"nodeId": "country", "parentId": "root", "ignored": false, "backendDOMNodeId": 40,
               "role": {"value": "combobox"}, "name": {"value": "País ou região"},
               "value": {"value": value}, "properties": properties})
    };
    let root = |children: &[&str]| json!({"nodeId": "root", "ignored": false, "role": {"value": "RootWebArea"}, "childIds": children});
    vec![
        json!({"nodes": [root(&["country"]), trigger("Selecionar", false, false)]}),
        json!({"nodes": [
            root(&["country", "list"]),
            trigger("Selecionar", true, true),
            {"nodeId": "list", "parentId": "root", "ignored": false, "backendDOMNodeId": 60,
             "role": {"value": "listbox"}, "childIds": ["us", "uk"]},
            {"nodeId": "us", "parentId": "list", "ignored": false, "backendDOMNodeId": 61,
             "role": {"value": "option"}, "name": {"value": "English (US)"}},
            {"nodeId": "uk", "parentId": "list", "ignored": false, "backendDOMNodeId": 62,
             "role": {"value": "option"}, "name": {"value": "English (UK)"}}
        ]}),
        json!({"nodes": [root(&["country"]), trigger("English (UK)", false, false)]}),
    ]
}

async fn choose_in_aria_dropdown(trusted_input_limited: bool) -> (Fixture, Value) {
    let f = fixture_with_platform(
        |st| {
            st.select_page = true;
            st.native_select_reply = json!({ "kind": "none" });
            st.aria_trees = aria_dropdown_trees();
        },
        trusted_input_limited,
    )
    .await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot(&f, &target, &tab).await;
    let country = semantic_ref_named(&snap, "País ou região");
    let result = select_option(
        &f,
        &target,
        &tab,
        country["ref"].as_str().unwrap(),
        "english (uk)",
    )
    .await;
    let s = structured(&result).clone();
    (f, s)
}

#[tokio::test]
async fn custom_dropdowns_are_opened_and_chosen_through_the_accessibility_tree() {
    let (f, s) = choose_in_aria_dropdown(false).await;
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["route"], "accessibility");
    assert_eq!(s["click_route"], "cdp_input");
    assert_eq!(s["label"], "English (UK)");
    assert_eq!(s["effect"], "verified");
    let clicks: Vec<Value> = recorded_calls(&f, "Input.dispatchMouseEvent")
        .into_iter()
        .filter(|(_, params)| params["type"] == "mousePressed")
        .map(|(_, params)| json!([params["x"], params["y"]]))
        .collect();
    // O gatilho (40) e depois a opção (62), cada um no centro da sua caixa.
    assert_eq!(clicks, vec![json!([410.0, 405.0]), json!([630.0, 625.0])]);
    for method in ["Page.bringToFront", "Target.activateTarget"] {
        assert!(recorded_calls(&f, method).is_empty(), "{method} was sent");
    }
}

#[tokio::test]
async fn custom_dropdowns_use_page_events_where_cdp_input_would_raise_the_window() {
    let (f, s) = choose_in_aria_dropdown(true).await;
    assert_eq!(s["status"], "ok", "{s}");
    assert_eq!(s["click_route"], "dom_events");
    assert_eq!(s["effect"], "verified");
    let gestures = recorded_calls(&f, "Runtime.callFunctionOn")
        .into_iter()
        .filter(|(_, params)| params["userGesture"] == true)
        .map(|(_, params)| params["objectId"].clone())
        .collect::<Vec<_>>();
    assert_eq!(gestures, vec![json!("obj-40"), json!("obj-62")]);
    no_input_or_activation(&f);
}

#[tokio::test]
async fn select_refuses_refs_that_are_not_dropdowns_before_touching_the_page() {
    let f = fixture_with(|st| st.select_page = true).await;
    let (target, tab) = bind(&f).await;
    let snap = semantic_snapshot(&f, &target, &tab).await;
    let next = semantic_ref_named(&snap, "Seguinte");
    assert!(!next["actions"]
        .as_array()
        .unwrap()
        .contains(&json!("select")));
    let result = select_option(&f, &target, &tab, next["ref"].as_str().unwrap(), "Portugal").await;
    assert_eq!(
        structured(&result)["refusal"]["code"],
        "browser_action_unavailable"
    );
    assert!(recorded_calls(&f, "Runtime.callFunctionOn").is_empty());
}

// ── Dropdowns num Chromium real (opcional) ───────────────────────────────────

/// Corre contra um Chromium headless já aberto na página de dropdowns do
/// smoke (CUA_SELECT_SMOKE_WS = o WebSocket do browser), com Input do CDP e
/// depois com o recurso de eventos na página.
#[tokio::test]
#[ignore = "needs CUA_SELECT_SMOKE_WS: a headless Chromium on the dropdown fixture page"]
async fn dropdown_fixture_in_a_real_chromium() {
    let ws_url = std::env::var("CUA_SELECT_SMOKE_WS").expect("CUA_SELECT_SMOKE_WS");
    let page = super::cdp_ws::CdpConnection::connect(&ws_url)
        .await
        .unwrap();
    let targets = page
        .call(None, "Target.getTargets", json!({}))
        .await
        .unwrap();
    let target_id = targets["targetInfos"]
        .as_array()
        .unwrap()
        .iter()
        .find(|target| target["type"] == "page")
        .unwrap()["targetId"]
        .as_str()
        .unwrap()
        .to_owned();
    let attached = page
        .call(
            None,
            "Target.attachToTarget",
            json!({ "targetId": target_id, "flatten": true }),
        )
        .await
        .unwrap();
    let page_session = attached["sessionId"].as_str().unwrap().to_owned();
    let read = |expression: &'static str| {
        let page = &page;
        let page_session = page_session.clone();
        async move {
            page.call(
                Some(&page_session),
                "Runtime.evaluate",
                json!({ "expression": expression, "returnByValue": true }),
            )
            .await
            .unwrap()["result"]["value"]
                .clone()
        }
    };
    let cases: &[(&str, &str, &str, &str)] = &[
        ("Língua", "inglês (reino unido)", "plain.value", "en-GB"),
        (
            "País ou região",
            "Portugal",
            "document.getElementById('react-country-state').textContent",
            "PO2",
        ),
        (
            "Idioma principal",
            "Inglês (Reino Unido)",
            "document.getElementById('lang').textContent",
            "Inglês (Reino Unido)",
        ),
        (
            "Cidade",
            "Braga",
            "document.getElementById('city-state').textContent",
            "Braga",
        ),
        (
            "Ordenar",
            "Nome (A-Z)",
            "document.getElementById('sort').textContent",
            "Nome (A-Z)",
        ),
        (
            "Fuso horário",
            "Açores",
            "document.getElementById('tz').textContent",
            "Açores (AZOT)",
        ),
        (
            "Cor",
            "Verde",
            "document.getElementById('shadow-host').shadowRoot.getElementById('t').textContent",
            "Verde",
        ),
        (
            "Produto",
            "Artigo 873",
            "document.getElementById('product').textContent",
            "Artigo 873",
        ),
    ];
    let mut failures = Vec::new();
    for limited in [false, true] {
        page.call(Some(&page_session), "Page.reload", json!({}))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let engine = BrowserEngine::new(Arc::new(FixturePlatform {
            ws_url: ws_url.clone(),
            trusted_input_limited: limited,
            managed_endpoint_visible: true,
            process_role: BrowserProcessRole::EmbeddedApplication,
            managed_discovery_invoked: Arc::new(AtomicBool::new(false)),
            existing_endpoint_visible: Arc::new(AtomicBool::new(true)),
            setup_invoked: Arc::new(AtomicBool::new(false)),
            setup_aborted: Arc::new(AtomicBool::new(false)),
            stall_consent: false,
        }));
        let bound = GetBrowserStateTool::new(engine.clone())
            .invoke(json!({ "pid": 1, "window_id": 7, "session": SESSION }))
            .await;
        let bound = structured(&bound).clone();
        assert_eq!(bound["status"], "ok", "bind: {bound}");
        let target = bound["target_id"].as_str().unwrap().to_owned();
        let tab = bound["tabs"][0]["tab_id"].as_str().unwrap().to_owned();
        for (name, option, check, expected) in cases {
            let snap = GetBrowserStateTool::new(engine.clone())
                .invoke(json!({
                    "target_id": target, "tab_id": tab, "session": SESSION,
                    "snapshot_format": "semantic_v2", "query": name,
                }))
                .await;
            let snap = structured(&snap).clone();
            let Some(found) = snap["refs"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|entry| entry["name"] == *name || entry["value"] == *name)
            else {
                println!("FAIL limited={limited} {name}: no ref in {}", snap["refs"]);
                failures.push(format!("limited={limited} {name}: no ref"));
                continue;
            };
            let result = BrowserSelectTool::new(engine.clone())
                .invoke(json!({
                    "target_id": target, "tab_id": tab, "session": SESSION,
                    "ref": found["ref"], "option": option,
                }))
                .await;
            let s = result
                .structured_content
                .clone()
                .unwrap_or_else(|| json!({ "error": format!("{:?}", result.content) }));
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            let shown = read(check).await;
            let pass = s["status"] == "ok" && s["effect"] == "verified" && shown == *expected;
            println!(
                "{} limited={limited} {name} -> {option}: route={} click_route={} effect={} page={shown}",
                if pass { "PASS" } else { "FAIL" },
                s["route"], s["click_route"], s["effect"]
            );
            if !pass {
                println!(
                    "  detail: {s} actions={} role={}",
                    found["actions"], found["role"]
                );
                failures.push(format!("limited={limited} {name}"));
            }
        }
        let snap = GetBrowserStateTool::new(engine.clone())
            .invoke(json!({ "target_id": target, "tab_id": tab, "session": SESSION, "snapshot_format": "semantic_v2", "query": "Açores (AZOT)" }))
            .await;
        let sort_ref = structured(&snap)["refs"][0]["ref"].clone();
        let missing = BrowserSelectTool::new(engine.clone())
            .invoke(json!({
                "target_id": target, "tab_id": tab, "session": SESSION,
                "ref": sort_ref, "option": "Klingon",
            }))
            .await;
        let s = missing.structured_content.clone().unwrap_or_default();
        let closed = read("document.querySelector('[aria-label=Fusos]') === null").await;
        println!(
            "no-match limited={limited}: {} closed={closed}",
            s["refusal"]["message"]
        );
        if s["refusal"]["code"] != "browser_action_unavailable" || closed != true {
            failures.push(format!("limited={limited} no-match: {s} closed={closed}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
