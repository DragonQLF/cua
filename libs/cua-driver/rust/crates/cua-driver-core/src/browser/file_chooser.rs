//! Native file-chooser containment for driver-delivered input.
//!
//! Chromium shows its native file picker only when no DevTools client
//! intercepts the request. Driver input that can carry user activation runs
//! inside a [`ChooserInterception`], so a page can never open a native window
//! over the user's desktop on the driver's behalf; the request is reported
//! instead, and `browser_set_input_files` fills it through CDP.
//!
//! Measured on Chromium 153 (macOS):
//! - interception and `Page.fileChooserOpened` require the Page domain on the
//!   intercepting session; without it the native picker still opens;
//! - interception belongs to a frame agent: a tab session does not cover an
//!   out-of-process iframe, so every OOPIF child is attached and intercepted
//!   too;
//! - File System Access pickers (`showOpenFilePicker()`,
//!   `showSaveFilePicker()`, `showDirectoryPicker()`) are intercepted as well:
//!   the page's promise rejects with `AbortError` and the event names no
//!   element, so there is nothing CDP can fill;
//! - an intercepted chooser left unanswered does not block later choosers,
//!   and once interception ends the user's own clicks open the picker again.
//!
//! The interception uses its own tab attachment for the interval and detaches
//! it afterwards, so no interception state outlives the tool call.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::protocol::{Content, ToolResult};

use super::cdp_ws::{CdpConnection, CdpEvent};
use super::engine::{is_method_unsupported, BrowserEngine, ValidatedTab};

/// How long a guarded action waits for a chooser its input requested. The
/// browser reports the request asynchronously after the input is acked.
pub(crate) const CHOOSER_SETTLE: Duration = Duration::from_millis(150);

/// How long `browser_set_input_files` waits for the chooser its own
/// activation is expected to open.
pub(crate) const CHOOSER_WAIT: Duration = Duration::from_millis(2500);

/// Bounds the OOPIF walk so a hostile page cannot make one click attach an
/// unbounded number of frames.
const MAX_INTERCEPTED_FRAMES: usize = 32;
const MAX_FRAME_DEPTH: usize = 3;

/// One chooser a page requested while interception was active.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpenedChooser {
    /// The intercepting session the request arrived on; the element's
    /// `backendNodeId` is valid there.
    pub session_id: String,
    /// The `<input type=file>` that asked. `None` for File System Access
    /// pickers, which have no element.
    pub backend_node_id: Option<i64>,
    pub multiple: bool,
}

pub(crate) struct ChooserInterception {
    conn: Arc<CdpConnection>,
    tab_session: String,
    /// Sessions with Page enabled and interception on, tab first.
    intercepting: Vec<String>,
    /// Sessions on which this interception enabled auto-attach.
    auto_attached: Vec<String>,
    /// OOPIF child sessions this interception attached.
    children: Vec<String>,
    events: UnboundedReceiver<CdpEvent>,
    opened: Vec<OpenedChooser>,
}

enum EnableOutcome {
    Enabled,
    Unsupported,
}

impl ChooserInterception {
    /// Start intercepting every frame of the exact tab. `Ok(None)` means the
    /// browser does not implement chooser interception at all.
    ///
    /// `cancel` asks Chromium to cancel an intercepted chooser right away
    /// (the page sees the `cancel` event a dismissed picker fires); leave it
    /// off when the chooser is about to be filled.
    pub(crate) async fn begin(
        engine: &BrowserEngine,
        validated: &ValidatedTab,
        cancel: bool,
    ) -> Result<Option<Self>, String> {
        let conn = validated.conn.clone();
        // Subscribe before attaching so no chooser event can be missed.
        let events = conn.subscribe();
        let tab_session = engine
            .attach(&conn, &validated.tab.cdp_target_id)
            .await
            .map_err(|refusal| refusal.message)?;
        let mut this = Self {
            conn,
            tab_session: tab_session.clone(),
            intercepting: Vec::new(),
            auto_attached: Vec::new(),
            children: Vec::new(),
            events,
            opened: Vec::new(),
        };
        match this.enable(&tab_session, cancel).await {
            Ok(EnableOutcome::Enabled) => {}
            Ok(EnableOutcome::Unsupported) => {
                this.release().await;
                return Ok(None);
            }
            Err(error) => {
                this.release().await;
                return Err(error);
            }
        }

        let mut frontier = vec![(tab_session, 0usize)];
        while let Some((parent, depth)) = frontier.pop() {
            if depth >= MAX_FRAME_DEPTH {
                continue;
            }
            // Auto-attach stays on for the interval: turning it off would
            // detach the very children that are intercepting.
            let attached = engine.attached_iframe_children(&this.conn, &parent).await;
            let Ok(children) = attached else {
                continue; // no OOPIF support on this session: nothing to cover
            };
            this.auto_attached.push(parent);
            for child in children {
                if this.children.len() >= MAX_INTERCEPTED_FRAMES {
                    break;
                }
                this.children.push(child.session_id.clone());
                // A child that fails here is being torn down with its frame,
                // and a frame that is going away cannot open a chooser.
                if matches!(
                    this.enable(&child.session_id, cancel).await,
                    Ok(EnableOutcome::Enabled)
                ) {
                    frontier.push((child.session_id, depth + 1));
                }
            }
        }
        Ok(Some(this))
    }

    async fn enable(&mut self, session: &str, cancel: bool) -> Result<EnableOutcome, String> {
        for (method, params) in [
            ("Page.enable", json!({})),
            (
                "Page.setInterceptFileChooserDialog",
                json!({ "enabled": true, "cancel": cancel }),
            ),
        ] {
            match self.conn.call(Some(session), method, params).await {
                Ok(_) => {}
                Err(error) if is_method_unsupported(&error) => {
                    return Ok(EnableOutcome::Unsupported)
                }
                Err(error) => return Err(error.to_string()),
            }
            if method == "Page.enable" {
                // Recorded before interception so release always disables Page.
                self.intercepting.push(session.to_owned());
            }
        }
        Ok(EnableOutcome::Enabled)
    }

    fn record(&mut self, event: CdpEvent) -> Option<OpenedChooser> {
        if event.method != "Page.fileChooserOpened" {
            return None;
        }
        let session_id = event.session_id?;
        if !self.intercepting.contains(&session_id) {
            return None;
        }
        let chooser = OpenedChooser {
            session_id,
            backend_node_id: event.params.get("backendNodeId").and_then(Value::as_i64),
            multiple: event.params.get("mode").and_then(Value::as_str) == Some("selectMultiple"),
        };
        self.opened.push(chooser.clone());
        Some(chooser)
    }

    /// The next chooser requested since the last call, waiting at most
    /// `timeout` for one.
    pub(crate) async fn next_chooser(&mut self, timeout: Duration) -> Option<OpenedChooser> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let event = match tokio::time::timeout_at(deadline, self.events.recv()).await {
                Ok(Some(event)) => event,
                Ok(None) | Err(_) => return None,
            };
            if let Some(chooser) = self.record(event) {
                return Some(chooser);
            }
        }
    }

    /// End interception and return every chooser the page requested. When
    /// none has arrived yet, wait up to `settle` for a late report first.
    pub(crate) async fn finish(mut self, settle: Duration) -> Vec<OpenedChooser> {
        if self.opened.is_empty() {
            let _ = self.next_chooser(settle).await;
        }
        while let Ok(event) = self.events.try_recv() {
            let _ = self.record(event);
        }
        self.release().await;
        std::mem::take(&mut self.opened)
    }

    async fn release(&mut self) {
        let conn = self.conn.clone();
        for session in self.intercepting.iter().rev() {
            let _ = conn
                .call(
                    Some(session),
                    "Page.setInterceptFileChooserDialog",
                    json!({ "enabled": false }),
                )
                .await;
            let _ = conn.call(Some(session), "Page.disable", json!({})).await;
        }
        for session in self.auto_attached.iter().rev() {
            let _ = conn
                .call(
                    Some(session),
                    "Target.setAutoAttach",
                    json!({ "autoAttach": false, "waitForDebuggerOnStart": false, "flatten": true }),
                )
                .await;
        }
        // Detaching drops whatever interception state a failed disable left
        // behind, so the user's own clicks always get their picker back.
        for session in self
            .children
            .iter()
            .rev()
            .chain(std::iter::once(&self.tab_session))
        {
            let _ = conn
                .call(
                    None,
                    "Target.detachFromTarget",
                    json!({ "sessionId": session }),
                )
                .await;
        }
        self.intercepting.clear();
        self.auto_attached.clear();
        self.children.clear();
    }
}

/// Run one input delivery with every native file chooser contained, and
/// report any chooser the page requested on the delivery's result.
pub(crate) async fn deliver_with_chooser_contained<F>(
    engine: &BrowserEngine,
    validated: &ValidatedTab,
    ext_ref: Option<&str>,
    delivery: F,
) -> ToolResult
where
    F: std::future::Future<Output = ToolResult>,
{
    let interception = match ChooserInterception::begin(engine, validated, true).await {
        Ok(interception) => interception,
        Err(error) => {
            return super::refusal::BrowserRefusal::new(
                super::refusal::BrowserRefusalCode::BrowserInputTrustUnavailable,
                format!(
                    "the page's native file picker could not be contained, so no input was delivered: {error}"
                ),
            )
            .with_detail(json!({ "delivery": "not_attempted", "retryable": true }))
            .to_tool_result();
        }
    };
    let result = delivery.await;
    let opened = match interception {
        Some(interception) => interception.finish(CHOOSER_SETTLE).await,
        None => Vec::new(),
    };
    report_contained_choosers(result, &opened, ext_ref)
}

/// Tell the caller that its input asked for a file chooser and how to
/// complete it, without ever showing the native picker.
pub(crate) fn report_contained_choosers(
    mut result: ToolResult,
    opened: &[OpenedChooser],
    ext_ref: Option<&str>,
) -> ToolResult {
    if opened.is_empty() {
        return result;
    }
    let fillable = opened
        .iter()
        .any(|chooser| chooser.backend_node_id.is_some());
    let multiple = opened.iter().any(|chooser| chooser.multiple);
    let control = ext_ref
        .map(|reference| format!("ref {reference}"))
        .unwrap_or_else(|| "the ref of the control that asked".to_owned());
    let note = if fillable {
        format!(
            "The page asked for a file chooser; no native picker was shown. To upload, call browser_set_input_files with {control} and the files."
        )
    } else {
        format!(
            "The page asked for a File System Access picker; no native picker was shown and the page received an AbortError. CDP cannot fill that picker; browser_set_input_files with {control} can still drop the files on the control."
        )
    };
    match result.content.iter_mut().find_map(|content| match content {
        Content::Text { text, .. } => Some(text),
        _ => None,
    }) {
        Some(text) => {
            text.push_str(if text.ends_with('.') { " " } else { ". " });
            text.push_str(&note);
        }
        None => result.content.push(Content::text(note.clone())),
    }
    let report = json!({
        "requested": true,
        "shown": false,
        "fillable": fillable,
        "multiple": multiple,
        "next_action": "browser_set_input_files",
    });
    match result.structured_content.as_mut() {
        Some(Value::Object(structured)) => {
            // Action tools publish a closed contract; there the next step
            // travels as a page escalation and in the summary text.
            if structured.get("status").and_then(Value::as_str) == Some("ok")
                && !structured.contains_key("escalation")
            {
                structured.insert(
                    "escalation".into(),
                    json!({ "recommended": "page", "reason": note }),
                );
            }
            structured.insert("file_chooser".into(), report);
        }
        _ => result.structured_content = Some(json!({ "file_chooser": report })),
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chooser(backend: Option<i64>, multiple: bool) -> OpenedChooser {
        OpenedChooser {
            session_id: "s".into(),
            backend_node_id: backend,
            multiple,
        }
    }

    #[test]
    fn results_without_a_chooser_are_unchanged() {
        let result = ToolResult::text("clicked").with_structured(json!({ "status": "ok" }));
        let reported = report_contained_choosers(result, &[], Some("p1:2"));
        assert!(reported
            .structured_content
            .unwrap()
            .get("file_chooser")
            .is_none());
    }

    #[test]
    fn a_contained_chooser_points_to_browser_set_input_files() {
        let result =
            ToolResult::text("clicked (10, 20) in tab1").with_structured(json!({ "status": "ok" }));
        let reported = report_contained_choosers(result, &[chooser(Some(7), true)], Some("p1:2"));
        let Content::Text { text, .. } = &reported.content[0] else {
            panic!("text content");
        };
        assert!(text.contains("no native picker was shown"), "{text}");
        assert!(
            text.contains("browser_set_input_files with ref p1:2"),
            "{text}"
        );
        let report = &reported.structured_content.as_ref().unwrap()["file_chooser"];
        assert_eq!(report["shown"], false);
        assert_eq!(report["fillable"], true);
        assert_eq!(report["multiple"], true);
        assert_eq!(report["next_action"], "browser_set_input_files");
        let structured = reported.structured_content.as_ref().unwrap();
        assert_eq!(structured["escalation"]["recommended"], "page");
    }

    #[test]
    fn the_closed_action_contract_still_points_to_a_page_action() {
        let legacy = json!({ "status": "ok", "route": "trusted", "x": 1.0, "y": 2.0 });
        let result = ToolResult::text("clicked").with_structured(legacy);
        let reported = report_contained_choosers(result, &[chooser(Some(7), false)], Some("p1:2"));
        let record = crate::action_record::ActionExecutionRecord::from_legacy(
            "browser_click",
            &json!({ "input_route": "trusted" }),
            reported.structured_content.as_ref().unwrap(),
        )
        .expect("browser_click records an action");
        let public = record.public_result().expect("valid public result");
        let escalation = public.escalation.expect("escalation survives projection");
        assert_eq!(
            escalation.target,
            cua_driver_contract::ActionEscalationTarget::Page
        );
    }

    #[test]
    fn a_file_system_access_picker_is_reported_as_unfillable() {
        let result = ToolResult::text("clicked");
        let reported = report_contained_choosers(result, &[chooser(None, false)], None);
        let Content::Text { text, .. } = &reported.content[0] else {
            panic!("text content");
        };
        assert!(text.contains("File System Access picker"), "{text}");
        assert_eq!(
            reported.structured_content.unwrap()["file_chooser"]["fillable"],
            false
        );
    }
}
