//! `browser_select`: escolher uma opção num dropdown sem rato nem teclado reais.
//!
//! O menu de um `<select>` nativo é desenhado pelo browser fora da página
//! (no macOS, um NSMenu): o CDP não o vê nem o opera, e um clique numa
//! `<option>` não escolhe nada. Por isso a opção escolhe-se pela própria
//! página, como o `selectOption` do Playwright e o `fill` do Chrome DevTools
//! MCP. Os dropdowns feitos à mão operam-se como no DevTools MCP: pela árvore
//! de acessibilidade e cliques na caixa dos elementos.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::protocol::ToolResult;
use crate::tool::{ProtectedResourceOwnership, Tool, ToolDef};
use crate::tool_args::ArgsExt;

use super::cdp_ws::CdpConnection;
use super::engine::BrowserEngine;
use super::platform::BrowserVisualActionKind;
use super::refusal::{BrowserRefusal, BrowserRefusalCode};
use super::session_schema as schema_session;
use super::store::BrowserActionKind;
use super::tools::{
    browser_protected_resource_scope, browser_resource_ownership, enter_focus_emulation,
    parse_delivery_mode, quad_center, select_all_in_element, trusted_click_on_node,
};

/// Avaliada no nó da ref. Rótulo primeiro, depois valor; exato (sem
/// maiúsculas nem espaços a mais) antes de parcial, que tem de ser único.
const MATCH_OPTION: &str = r#"
    const norm = (s) => String(s == null ? '' : s).replace(/\s+/g, ' ').trim().toLowerCase();
    const pick = (items, labelOf, valueOf, wanted) => {
        const w = norm(wanted);
        if (!w) return { error: 'no_option' };
        const usable = (list) => { const on = list.filter((o) => !o.disabled && o.getAttribute('aria-disabled') !== 'true'); return on.length ? on : list; };
        let found = usable(items.filter((o) => norm(labelOf(o)) === w));
        if (!found.length) found = usable(items.filter((o) => norm(valueOf(o)) === w));
        if (!found.length) found = usable(items.filter((o) => norm(labelOf(o)).includes(w)));
        if (found.length === 1) return { option: found[0] };
        if (!found.length) return { error: 'no_match' };
        return { error: 'ambiguous', matches: found.slice(0, 12).map(labelOf) };
    };
"#;

const NATIVE_SELECT: &str = r#"function(wanted) {
    __MATCH__
    const labelOf = (o) => String(o.label || o.textContent || '').replace(/\s+/g, ' ').trim();
    const isSelect = (n) => !!n && n.localName === 'select';
    let select = null;
    let option = null;
    if (this.localName === 'option') { select = this.closest('select'); option = this; }
    else if (isSelect(this)) select = this;
    else if (this.localName === 'label' && isSelect(this.control)) select = this.control;
    else if (this.querySelectorAll) {
        const inside = this.querySelectorAll('select');
        if (inside.length === 1) select = inside[0];
    }
    if (!select) return { kind: 'none' };
    const options = Array.from(select.options);
    const listed = { options: options.slice(0, 40).map(labelOf), options_total: options.length };
    if (!option) {
        const picked = pick(options, labelOf, (o) => o.value, wanted);
        if (!picked.option) return Object.assign({ kind: 'native' }, listed, picked);
        option = picked.option;
    }
    if (select.disabled) return { kind: 'native', error: 'disabled' };
    if (option.disabled) return { kind: 'native', error: 'option_disabled', label: labelOf(option) };
    const before = select.selectedIndex;
    // Como o Playwright: a opção muda pela página e os eventos que um clique
    // humano geraria avisam React/Vue/Angular, que ouvem input e change.
    option.selected = true;
    select.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
    select.dispatchEvent(new Event('change', { bubbles: true }));
    const now = select.options[select.selectedIndex];
    return {
        kind: 'native',
        label: labelOf(option),
        value: option.value,
        changed: before !== option.index,
        kept: option.selected,
        now: now ? labelOf(now) : null,
    };
}"#;

fn native_select_js() -> String {
    NATIVE_SELECT.replace("__MATCH__", MATCH_OPTION)
}

pub struct BrowserSelectTool {
    def: ToolDef,
    engine: Arc<BrowserEngine>,
}

impl BrowserSelectTool {
    pub fn new(engine: Arc<BrowserEngine>) -> Self {
        Self {
            def: ToolDef {
                name: "browser_select".into(),
                description: "Choose an option in a dropdown of an exactly-bound tab, by ref. Use it for every dropdown instead of browser_click or keys: a native <select> (its menu is drawn by the browser outside the page, so clicks and keys cannot operate it) is set through the page and gets the input and change events a person's choice would fire. Any other dropdown (combobox, listbox, menu button, also in shadow DOM or rendered elsewhere in the page) is operated through the accessibility tree: the trigger is clicked open, the option is found by its accessible name (typing into a searchable combobox and scrolling a long list when needed) and clicked, then the trigger is checked for the choice. option matches labels, then values: exact (ignoring case and extra spaces) first, then a unique partial match. get_browser_state (semantic_v2) lists a native select's options on its ref; no match returns the options seen. The ref may also be the option itself, without option. Refused for heuristic bindings.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "target_id": {
                            "type": "string",
                            "description": "Opaque browser target id minted by get_browser_state (session-scoped; never a CDP id)."
                        },
                        "tab_id": {
                            "type": "string",
                            "description": "Opaque tab id from get_browser_state (session-scoped)."
                        },
                        "session": schema_session(),
                        "ref": {
                            "type": "string",
                            "description": "Ref of the dropdown (a select, combobox, listbox or menu button, or its label or wrapper), or of one of its options, from get_browser_state. Refs are invalidated by navigation and by newer snapshots of the same tab."
                        },
                        "option": {
                            "type": "string",
                            "description": "Label or value of the option to choose, as shown on the page (e.g. \"Portugal\"). Omit only when ref is the option itself."
                        },
                        "delivery_mode": {
                            "type": "string",
                            "enum": ["background", "foreground"],
                            "default": "background",
                            "description": "Only for dropdowns that are not a native select. background (default) clicks with trusted CDP input where that keeps the browser window in the background, and otherwise with synthetic in-page pointer events. foreground accepts that trusted input may activate the browser window."
                        }
                    },
                    "required": ["target_id", "tab_id", "ref"],
                    "additionalProperties": true
                }),
                read_only: false,
                destructive: false,
                idempotent: true,
                open_world: true,
            },
            engine,
        }
    }
}

fn explicit_session(args: &Value) -> Result<String, ToolResult> {
    let session = args
        .opt_str("session")
        .or_else(|| args.opt_str("_session_id"))
        .unwrap_or_else(|| "default".into());
    if session.is_empty() || session == "default" {
        return Err(ToolResult::error(
            "Browser targets and page refs are session-scoped capabilities — declare an \
             explicit session (start_session) and pass its id on this call.",
        ));
    }
    Ok(session)
}

fn unavailable(message: String, detail: Value) -> ToolResult {
    BrowserRefusal::new(BrowserRefusalCode::BrowserActionUnavailable, message)
        .with_detail(detail)
        .to_tool_result()
}

fn quoted_list(items: Option<&Value>) -> String {
    items
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|label| format!("{label:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default()
}

/// O que a função devolveu, ou o erro do CDP / da página.
async fn call_on_node(
    conn: &CdpConnection,
    cdp: &str,
    object_id: &str,
    function: &str,
    option: &Value,
    await_promise: bool,
) -> Result<Value, String> {
    let reply = conn
        .call(
            Some(cdp),
            "Runtime.callFunctionOn",
            json!({
                "objectId": object_id,
                "functionDeclaration": function,
                "arguments": [{ "value": option }],
                "returnByValue": true,
                "awaitPromise": await_promise,
            }),
        )
        .await
        .map_err(|error| error.to_string())?;
    if let Some(exception) = reply.get("exceptionDetails") {
        return Err(exception
            .pointer("/exception/description")
            .or_else(|| exception.get("text"))
            .and_then(Value::as_str)
            .unwrap_or("the page threw")
            .to_owned());
    }
    Ok(reply
        .pointer("/result/value")
        .cloned()
        .unwrap_or(Value::Null))
}

#[async_trait]
impl Tool for BrowserSelectTool {
    fn def(&self) -> &ToolDef {
        &self.def
    }

    async fn protected_resource_ownership(
        &self,
        adapter_id: &str,
        args: &Value,
    ) -> ProtectedResourceOwnership {
        if adapter_id == "browser_bound_input" {
            browser_resource_ownership(&self.engine, args)
        } else {
            ProtectedResourceOwnership::UserOwned
        }
    }

    async fn protected_resource_scope(
        &self,
        adapter_id: &str,
        args: &Value,
    ) -> Result<Option<Value>, String> {
        if adapter_id == "browser_bound_input" {
            browser_protected_resource_scope(&self.engine, args, "browser_select").await
        } else {
            Ok(None)
        }
    }

    async fn invoke(&self, args: Value) -> ToolResult {
        let (target_id, tab_id, ext_ref) = match (
            args.require_str("target_id"),
            args.require_str("tab_id"),
            args.require_str("ref"),
        ) {
            (Ok(target), Ok(tab), Ok(ext_ref)) => (target, tab, ext_ref),
            (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => return error,
        };
        let option = match args.get("option") {
            None | Some(Value::Null) => Value::Null,
            Some(Value::String(text)) if !text.trim().is_empty() => Value::String(text.clone()),
            Some(_) => return ToolResult::error("option must be a non-empty string"),
        };
        let foreground = match parse_delivery_mode(&args) {
            Ok(foreground) => foreground,
            Err(error) => return ToolResult::error(error),
        };
        let session = match explicit_session(&args) {
            Ok(session) => session,
            Err(error) => return error,
        };
        let _mutation = match self
            .engine
            .lock_mutation(&session, &target_id, &tab_id)
            .await
        {
            Ok(guard) => guard,
            Err(refusal) => return refusal.to_tool_result(),
        };
        let validated = match self
            .engine
            .revalidate_for_mutation(&session, &target_id, Some(&tab_id))
            .await
        {
            Ok(validated) => validated,
            Err(refusal) => return refusal.to_tool_result(),
        };
        let entry = match self
            .engine
            .store
            .resolve_ref(&session, &target_id, &tab_id, &ext_ref)
        {
            Ok(entry) => entry,
            Err(refusal) => return refusal.to_tool_result(),
        };
        if entry.semantic && !entry.actions.contains(&BrowserActionKind::Select) {
            return BrowserRefusal::new(
                BrowserRefusalCode::BrowserActionUnavailable,
                format!("semantic ref {ext_ref} does not declare the select action: it is not a dropdown or an option"),
            )
            .to_tool_result();
        }
        let cdp_session = match self
            .engine
            .frame_session_for_mutation(&session, &target_id, &tab_id, &validated, &entry.frame)
            .await
        {
            Ok(session) => session,
            Err(refusal) => return refusal.to_tool_result(),
        };
        let conn = &validated.conn;
        let cdp = cdp_session.as_str();
        let Some(object_id) = conn
            .call(
                Some(cdp),
                "DOM.resolveNode",
                json!({ "backendNodeId": entry.backend_node_id }),
            )
            .await
            .ok()
            .and_then(|resolved| resolved["object"]["objectId"].as_str().map(str::to_owned))
        else {
            return BrowserRefusal::new(
                BrowserRefusalCode::BrowserRefStale,
                "the dropdown ref no longer resolves in the live page",
            )
            .to_tool_result();
        };

        // Só o cursor do agente: sem caixa, escolhe-se na mesma.
        if let Ok(box_model) = conn
            .call(
                Some(cdp),
                "DOM.getBoxModel",
                json!({ "backendNodeId": entry.backend_node_id }),
            )
            .await
        {
            if let Some((x, y)) = quad_center(&box_model) {
                self.engine
                    .visualize_browser_action(
                        &session,
                        &validated,
                        cdp,
                        x,
                        y,
                        BrowserVisualActionKind::Click,
                    )
                    .await;
            }
        }

        let wanted = option.as_str().map(str::to_owned);
        let native =
            match call_on_node(conn, cdp, &object_id, &native_select_js(), &option, false).await {
                Ok(outcome) => outcome,
                Err(error) => {
                    return ToolResult::error(format!("selecting through the page failed: {error}"))
                }
            };
        let ok = |route: &str,
                  click_route: Option<&str>,
                  label: &str,
                  verified: bool,
                  extra: Value| {
            let text = match (route, verified) {
                ("native_select", _) => format!(
                    "selected {label:?} in the dropdown {ext_ref} in {tab_id}, through the page; no native menu was opened"
                ),
                (_, true) => format!("opened the dropdown {ext_ref} in {tab_id} and chose {label:?}"),
                (_, false) => format!(
                    "opened the dropdown {ext_ref} in {tab_id} and clicked {label:?}, but the dropdown does not show it as chosen. Refresh page state and verify"
                ),
            };
            let mut structured = json!({
                "status": "ok",
                "target_id": target_id,
                "tab_id": tab_id,
                "ref": ext_ref,
                "frame": entry.frame.kind.as_str(),
                "route": route,
                "effect": if verified { "verified" } else { "unverifiable" },
                "label": label,
                "native_menu_shown": false,
            });
            if let Some(click_route) = click_route {
                structured["click_route"] = json!(click_route);
            }
            if let (Value::Object(target), Value::Object(extra)) = (&mut structured, extra) {
                target.extend(extra);
            }
            ToolResult::text(text).with_structured(structured)
        };
        let no_match = |kind: &str, options: &Value, total: Option<u64>| {
            let listed = quoted_list(Some(options));
            unavailable(
                format!(
                    "no option of this dropdown matches {:?}. Its options{}: {}",
                    wanted.as_deref().unwrap_or(""),
                    match total {
                        Some(total) if total > 40 => format!(" (first 40 of {total})"),
                        _ => String::new(),
                    },
                    if listed.is_empty() {
                        "none visible"
                    } else {
                        &listed
                    }
                ),
                json!({ "kind": kind, "options": options, "options_total": total }),
            )
        };
        let ambiguous = |kind: &str, matches: &Value| {
            unavailable(
                format!(
                    "{:?} matches several options: {}. Pass the exact label",
                    wanted.as_deref().unwrap_or(""),
                    quoted_list(Some(matches))
                ),
                json!({ "kind": kind, "matches": matches }),
            )
        };

        if native["kind"] == "native" {
            return match native["error"].as_str() {
                None if native["kept"] == false => unavailable(
                    format!(
                        "the page put the previous choice back after {:?} was selected; now showing {:?}",
                        native["label"].as_str().unwrap_or(""),
                        native["now"].as_str().unwrap_or("")
                    ),
                    json!({ "kind": "native", "reverted": true }),
                ),
                None => ok(
                    "native_select",
                    None,
                    native["label"].as_str().unwrap_or(""),
                    true,
                    json!({ "value": native["value"], "changed": native["changed"] }),
                ),
                Some("no_option") => {
                    ToolResult::error("option is required unless ref is the option itself")
                }
                Some("no_match") => no_match("native", &native["options"], native["options_total"].as_u64()),
                Some("ambiguous") => ambiguous("native", &native["matches"]),
                Some("disabled") => {
                    unavailable("the dropdown is disabled".into(), json!({ "kind": "native" }))
                }
                Some("option_disabled") => unavailable(
                    format!("the option {:?} is disabled", native["label"].as_str().unwrap_or("")),
                    json!({ "kind": "native" }),
                ),
                Some(other) => ToolResult::error(format!("browser_select failed: {other}")),
            };
        }

        // Onde o Input do CDP levantaria a janela do browser (macOS e Linux
        // num browser autónomo), clica-se com eventos na página.
        let trusted = foreground
            || validated.record.cdp_window_id.is_none()
            || self
                .engine
                .platform
                .standalone_trusted_input_background_limitation()
                .is_none();
        let page = AriaPage {
            conn,
            cdp,
            frame_id: entry
                .frame
                .identity
                .as_ref()
                .map(|identity| identity.frame_id.clone()),
            trusted,
        };
        let click_route = if trusted { "cdp_input" } else { "dom_events" };
        match page.select(entry.backend_node_id, wanted.as_deref()).await {
            Err(error) => ToolResult::error(format!("browser_select failed: {error}")),
            Ok(AriaOutcome::NotADropdown) => unavailable(
                format!("{ext_ref} is not a dropdown: no <select>, combobox, listbox or menu button was found at or inside it"),
                json!({ "kind": "none" }),
            ),
            Ok(AriaOutcome::NoOption) => {
                ToolResult::error("option is required unless ref is the option itself")
            }
            Ok(AriaOutcome::Ambiguous(matches)) => ambiguous("accessibility", &json!(matches)),
            Ok(AriaOutcome::NoMatch { options, opened }) => {
                if options.is_empty() && opened {
                    unavailable(
                        "the dropdown showed no options after it was clicked open".into(),
                        json!({ "kind": "accessibility", "click_route": click_route }),
                    )
                } else {
                    no_match("accessibility", &json!(options), None)
                }
            }
            Ok(AriaOutcome::Chosen { label, verified }) => ok(
                "accessibility",
                Some(click_route),
                &label,
                verified,
                json!({}),
            ),
        }
    }
}
// ── Dropdowns feitos à mão: pela árvore de acessibilidade ───────────────────
//
// Como o Chrome DevTools MCP: o gatilho e as opções encontram-se na árvore AX
// (atravessa shadow DOM e vê listas desenhadas em portais), clica-se na caixa
// do elemento e a lista aberta lê-se na árvore seguinte.

const OPTION_ROLES: &[&str] = &[
    "option",
    "listboxoption",
    "menuitem",
    "menuitemradio",
    "menuitemcheckbox",
    "treeitem",
];
const LIST_ROLES: &[&str] = &["listbox", "menu", "menubar", "tree", "grid", "treegrid"];
const OPEN_WAIT: Duration = Duration::from_millis(2_000);
const SEARCH_WAIT: Duration = Duration::from_millis(3_000);
const VERIFY_WAIT: Duration = Duration::from_millis(1_500);
const POLL: Duration = Duration::from_millis(150);
/// Listas virtualizadas: quantas vezes se rola e relê, e por quanto tempo.
const MAX_LIST_SCROLLS: usize = 200;
const LIST_SCROLL_BUDGET: Duration = Duration::from_secs(30);
const MAX_LISTED: usize = 40;

/// Clique de recurso, onde o Input do CDP levantaria a janela: a sequência
/// de eventos que um rato real geraria, com gesto do utilizador.
const DOM_CLICK: &str = r#"function() {
    const r = this.getBoundingClientRect();
    const at = { bubbles: true, cancelable: true, composed: true, view: window, button: 0, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 };
    this.dispatchEvent(new PointerEvent('pointerdown', Object.assign({ pointerType: 'mouse', isPrimary: true, buttons: 1 }, at)));
    this.dispatchEvent(new MouseEvent('mousedown', Object.assign({ buttons: 1 }, at)));
    if (this.focus) this.focus({ preventScroll: true });
    this.dispatchEvent(new PointerEvent('pointerup', Object.assign({ pointerType: 'mouse', isPrimary: true }, at)));
    this.dispatchEvent(new MouseEvent('mouseup', at));
    this.dispatchEvent(new MouseEvent('click', at));
}"#;

/// Rola uma página do contentor que rola a partir da lista (ela ou um
/// antepassado). Devolve se se mexeu.
const SCROLL_LIST: &str = r#"function() {
    let n = this;
    for (let i = 0; n && i < 8; i++, n = n.parentElement) {
        const s = getComputedStyle(n);
        if (n.scrollHeight > n.clientHeight + 1 && /(auto|scroll|overlay)/.test(s.overflowY)) {
            const before = n.scrollTop;
            n.scrollTop = before + Math.max(n.clientHeight * 0.9, 20);
            n.dispatchEvent(new Event('scroll'));
            return n.scrollTop !== before;
        }
    }
    return false;
}"#;

/// O que o gatilho mostra. Num campo vazio, o escolhido costuma estar ao lado;
/// o texto de uma lista aberta não conta.
const SHOWN_TEXT: &str = r#"function() {
    const squash = (t) => String(t || '').replace(/\s+/g, ' ').trim();
    const own = squash(this.value) || squash(this.innerText || this.textContent);
    if (own || this.localName !== 'input') return own;
    const outside = (box) => {
        const parts = [];
        const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            if (node.parentElement && !node.parentElement.closest('[role=listbox], [role=menu], [role=tree], [role=grid]')) parts.push(node.textContent);
        }
        return squash(parts.join(' '));
    };
    for (let box = this.parentElement, i = 0; box && i < 4; box = box.parentElement, i++) {
        const text = outside(box);
        if (text) return text;
    }
    return '';
}"#;

#[derive(Debug, Clone)]
struct AxNode {
    parent: Option<String>,
    children: Vec<String>,
    backend: Option<i64>,
    role: String,
    name: String,
    value: String,
    ignored: bool,
    properties: HashMap<String, Value>,
}

impl AxNode {
    fn property(&self, name: &str) -> Option<&Value> {
        self.properties.get(name)
    }

    fn flag(&self, name: &str) -> bool {
        self.property(name)
            .and_then(|value| value.get("value"))
            .is_some_and(|value| match value {
                Value::Bool(flag) => *flag,
                Value::String(text) => !text.is_empty() && text != "false",
                _ => false,
            })
    }

    fn related(&self, name: &str) -> Vec<i64> {
        self.property(name)
            .and_then(|value| value.get("relatedNodes"))
            .and_then(Value::as_array)
            .map(|nodes| {
                nodes
                    .iter()
                    .filter_map(|node| node.get("backendDOMNodeId").and_then(Value::as_i64))
                    .collect()
            })
            .unwrap_or_default()
    }
}

struct AxTree {
    order: Vec<String>,
    nodes: HashMap<String, AxNode>,
}

fn ax_text(value: Option<&Value>) -> String {
    match value.and_then(|value| value.get("value")) {
        Some(Value::String(text)) => text.split_whitespace().collect::<Vec<_>>().join(" "),
        Some(Value::Number(number)) => number.to_string(),
        _ => String::new(),
    }
}

fn parse_ax_tree(tree: &Value) -> AxTree {
    let mut order = Vec::new();
    let mut nodes = HashMap::new();
    for node in tree
        .get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(id) = node.get("nodeId").and_then(Value::as_str) else {
            continue;
        };
        let properties = node
            .get("properties")
            .and_then(Value::as_array)
            .map(|properties| {
                properties
                    .iter()
                    .filter_map(|property| {
                        Some((
                            property.get("name")?.as_str()?.to_owned(),
                            property.get("value")?.clone(),
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        order.push(id.to_owned());
        nodes.insert(
            id.to_owned(),
            AxNode {
                parent: node
                    .get("parentId")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                children: node
                    .get("childIds")
                    .and_then(Value::as_array)
                    .map(|ids| {
                        ids.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default(),
                backend: node.get("backendDOMNodeId").and_then(Value::as_i64),
                role: ax_text(node.get("role")).to_ascii_lowercase(),
                name: ax_text(node.get("name")),
                value: ax_text(node.get("value")),
                ignored: node.get("ignored").and_then(Value::as_bool) == Some(true),
                properties,
            },
        );
    }
    AxTree { order, nodes }
}

impl AxTree {
    fn by_backend(&self, backend: i64) -> Option<(&String, &AxNode)> {
        self.order
            .iter()
            .filter_map(|id| Some((id, self.nodes.get(id)?)))
            .filter(|(_, node)| node.backend == Some(backend))
            .min_by_key(|(_, node)| node.ignored)
    }

    fn ancestors<'a>(&'a self, id: &str) -> impl Iterator<Item = &'a AxNode> + 'a {
        let mut next = self.nodes.get(id).and_then(|node| node.parent.clone());
        std::iter::from_fn(move || {
            let node = self.nodes.get(next.as_deref()?)?;
            next = node.parent.clone();
            Some(node)
        })
    }

    /// Descendentes em ordem de documento (pré-ordem).
    fn descendants(&self, id: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = self
            .nodes
            .get(id)
            .map(|node| node.children.iter().rev().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        while let Some(next) = stack.pop() {
            if let Some(node) = self.nodes.get(&next) {
                stack.extend(node.children.iter().rev().cloned());
                out.push(next);
            }
        }
        out
    }

    fn is_option(&self, id: &str) -> bool {
        let Some(node) = self.nodes.get(id) else {
            return false;
        };
        if node.ignored || node.backend.is_none() {
            return false;
        }
        OPTION_ROLES.contains(&node.role.as_str())
            || (node.role == "gridcell"
                && self
                    .ancestors(id)
                    .any(|parent| matches!(parent.role.as_str(), "listbox" | "grid" | "treegrid")))
    }

    fn is_trigger(node: &AxNode) -> bool {
        if node.ignored || node.backend.is_none() {
            return false;
        }
        let popup = node
            .property("hasPopup")
            .and_then(|value| value.get("value"))
            .and_then(Value::as_str)
            .is_some_and(|value| value != "false");
        matches!(node.role.as_str(), "combobox" | "listbox" | "popupbutton") || popup
    }

    fn options_under(&self, id: &str) -> Vec<String> {
        self.descendants(id)
            .into_iter()
            .filter(|id| self.is_option(id))
            .collect()
    }

    fn option_backends(&self) -> HashSet<i64> {
        self.order
            .iter()
            .filter(|id| self.is_option(id))
            .filter_map(|id| self.nodes.get(id)?.backend)
            .collect()
    }

    /// As opções do gatilho: as da lista que ele controla ou possui; senão as
    /// que tem dentro; senão as que apareceram depois de o abrir.
    fn options_for(&self, host: i64, before: &HashSet<i64>) -> Vec<String> {
        let Some((host_id, host_node)) = self.by_backend(host) else {
            return Vec::new();
        };
        let mut found = Vec::new();
        for related in host_node
            .related("controls")
            .into_iter()
            .chain(host_node.related("owns"))
        {
            if let Some((list_id, _)) = self.by_backend(related) {
                found.extend(self.options_under(list_id));
            }
        }
        if found.is_empty() {
            found = self.options_under(host_id);
        }
        if found.is_empty() {
            found = self
                .order
                .iter()
                .filter(|id| self.is_option(id))
                .filter(|id| {
                    self.nodes[*id]
                        .backend
                        .is_some_and(|backend| !before.contains(&backend))
                })
                .cloned()
                .collect();
        }
        found
    }

    /// O contentor da lista onde estão as opções, para rolar uma lista
    /// virtualizada.
    fn list_of(&self, option: &str) -> Option<i64> {
        self.ancestors(option)
            .find(|node| LIST_ROLES.contains(&node.role.as_str()) && node.backend.is_some())
            .or_else(|| self.ancestors(option).find(|node| node.backend.is_some()))
            .and_then(|node| node.backend)
    }
}

fn norm(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

enum Picked {
    One(String),
    None,
    Several(Vec<String>),
}

/// Exato (sem maiúsculas nem espaços a mais) antes de parcial único.
fn pick_option(tree: &AxTree, candidates: &[String], wanted: &str) -> Picked {
    let wanted = norm(wanted);
    let label = |id: &String| norm(&tree.nodes[id].name);
    fn usable<'a>(tree: &AxTree, list: Vec<&'a String>) -> Vec<&'a String> {
        let enabled = list
            .iter()
            .copied()
            .filter(|id| !tree.nodes[*id].flag("disabled"))
            .collect::<Vec<_>>();
        if enabled.is_empty() {
            list
        } else {
            enabled
        }
    }
    let mut found = usable(
        tree,
        candidates
            .iter()
            .filter(|id| label(id) == wanted || norm(&tree.nodes[*id].value) == wanted)
            .collect(),
    );
    if found.is_empty() {
        found = usable(
            tree,
            candidates
                .iter()
                .filter(|id| label(id).contains(&wanted))
                .collect(),
        );
    }
    match found.len() {
        0 => Picked::None,
        1 => Picked::One(found[0].clone()),
        _ => Picked::Several(
            found
                .iter()
                .take(12)
                .map(|id| tree.nodes[*id].name.clone())
                .collect(),
        ),
    }
}

struct AriaPage<'a> {
    conn: &'a CdpConnection,
    cdp: &'a str,
    frame_id: Option<String>,
    /// Input do CDP onde não levanta a janela; senão eventos na página.
    trusted: bool,
}

enum AriaOutcome {
    NotADropdown,
    NoOption,
    Chosen { label: String, verified: bool },
    NoMatch { options: Vec<String>, opened: bool },
    Ambiguous(Vec<String>),
}

impl AriaPage<'_> {
    async fn tree(&self) -> Result<AxTree, String> {
        let params = match &self.frame_id {
            Some(frame_id) => json!({ "frameId": frame_id }),
            None => json!({}),
        };
        self.conn
            .call(Some(self.cdp), "Accessibility.getFullAXTree", params)
            .await
            .map(|tree| parse_ax_tree(&tree))
            .map_err(|error| format!("Accessibility.getFullAXTree failed: {error}"))
    }

    async fn object(&self, backend: i64) -> Option<String> {
        self.conn
            .call(
                Some(self.cdp),
                "DOM.resolveNode",
                json!({ "backendNodeId": backend }),
            )
            .await
            .ok()?
            .pointer("/object/objectId")?
            .as_str()
            .map(str::to_owned)
    }

    async fn run(&self, backend: i64, function: &str, user_gesture: bool) -> Option<Value> {
        let object_id = self.object(backend).await?;
        let reply = self
            .conn
            .call(
                Some(self.cdp),
                "Runtime.callFunctionOn",
                json!({
                    "objectId": object_id,
                    "functionDeclaration": function,
                    "returnByValue": true,
                    "userGesture": user_gesture,
                }),
            )
            .await
            .ok()?;
        if reply.get("exceptionDetails").is_some() {
            return None;
        }
        Some(
            reply
                .pointer("/result/value")
                .cloned()
                .unwrap_or(Value::Null),
        )
    }

    async fn click(&self, backend: i64) -> Result<(), String> {
        if self.trusted {
            return trusted_click_on_node(self.conn, self.cdp, backend)
                .await
                .map_err(|error| error.to_string());
        }
        let _ = self
            .conn
            .call(
                Some(self.cdp),
                "DOM.scrollIntoViewIfNeeded",
                json!({ "backendNodeId": backend }),
            )
            .await;
        self.run(backend, DOM_CLICK, true)
            .await
            .map(|_| ())
            .ok_or_else(|| "the element could not be clicked in the page".to_owned())
    }

    /// Escreve numa combobox de pesquisa como o browser_type: Input.insertText
    /// sob emulação de foco, que não levanta a janela.
    async fn type_search(&self, backend: i64, text: &str) -> Result<(), String> {
        let object_id = self
            .object(backend)
            .await
            .ok_or("the search field no longer resolves")?;
        enter_focus_emulation(self.conn, self.cdp, backend, &object_id).await?;
        let typed = async {
            select_all_in_element(self.conn, self.cdp, &object_id).await?;
            self.conn
                .call(Some(self.cdp), "Input.insertText", json!({ "text": text }))
                .await
                .map(|_| ())
                .map_err(|error| error.to_string())
        }
        .await;
        let _ = self
            .conn
            .call(
                Some(self.cdp),
                "Emulation.setFocusEmulationEnabled",
                json!({ "enabled": false }),
            )
            .await;
        typed
    }

    async fn escape(&self, backend: i64) {
        let _ = self
            .conn
            .call(
                Some(self.cdp),
                "Emulation.setFocusEmulationEnabled",
                json!({ "enabled": true }),
            )
            .await;
        let _ = self
            .conn
            .call(
                Some(self.cdp),
                "DOM.focus",
                json!({ "backendNodeId": backend }),
            )
            .await;
        for kind in ["keyDown", "keyUp"] {
            let _ = self
                .conn
                .call(
                    Some(self.cdp),
                    "Input.dispatchKeyEvent",
                    json!({ "type": kind, "key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27 }),
                )
                .await;
        }
        let _ = self
            .conn
            .call(
                Some(self.cdp),
                "Emulation.setFocusEmulationEnabled",
                json!({ "enabled": false }),
            )
            .await;
    }

    /// Fecha a lista que se abriu: Escape, e se continuar aberta, o gatilho
    /// outra vez, como uma pessoa faria.
    async fn close(&self, host: i64, before: &HashSet<i64>) {
        self.escape(host).await;
        tokio::time::sleep(POLL).await;
        let still_open = self
            .tree()
            .await
            .is_ok_and(|tree| !tree.options_for(host, before).is_empty());
        if still_open {
            let _ = self.click(host).await;
        }
    }

    async fn shown(&self, tree: &AxTree, backend: i64) -> String {
        let (name, value) = tree
            .by_backend(backend)
            .map(|(_, node)| (node.name.clone(), node.value.clone()))
            .unwrap_or_default();
        let text = self
            .run(backend, SHOWN_TEXT, false)
            .await
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_default();
        norm(&format!("{name} {value} {text}"))
    }

    async fn wait_options(
        &self,
        host: i64,
        before: &HashSet<i64>,
        wait: Duration,
    ) -> Result<(AxTree, Vec<String>), String> {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let tree = self.tree().await?;
            let options = tree.options_for(host, before);
            if !options.is_empty() || tokio::time::Instant::now() >= deadline {
                return Ok((tree, options));
            }
            tokio::time::sleep(POLL).await;
        }
    }

    async fn choose(
        &self,
        option: i64,
        label: String,
        host: Option<(i64, String)>,
    ) -> Result<AriaOutcome, String> {
        self.click(option).await?;
        let deadline = tokio::time::Instant::now() + VERIFY_WAIT;
        loop {
            tokio::time::sleep(POLL).await;
            let tree = self.tree().await?;
            let marked = tree
                .by_backend(option)
                .is_some_and(|(_, node)| node.flag("selected") || node.flag("checked"));
            let shows = match &host {
                Some((host, before)) => {
                    let now = self.shown(&tree, *host).await;
                    now != *before && now.contains(&norm(&label))
                }
                None => false,
            };
            if marked || shows || tokio::time::Instant::now() >= deadline {
                return Ok(AriaOutcome::Chosen {
                    label,
                    verified: marked || shows,
                });
            }
        }
    }

    async fn select(&self, reference: i64, wanted: Option<&str>) -> Result<AriaOutcome, String> {
        let tree = self.tree().await?;
        let Some((ref_id, ref_node)) = tree.by_backend(reference) else {
            return Ok(AriaOutcome::NotADropdown);
        };
        if tree.is_option(ref_id) {
            return self.choose(reference, ref_node.name.clone(), None).await;
        }
        // A ref pode ser o invólucro: o gatilho é ela ou o único lá dentro.
        let host = if AxTree::is_trigger(ref_node) {
            reference
        } else {
            let inner = tree
                .descendants(ref_id)
                .into_iter()
                .filter_map(|id| tree.nodes.get(&id))
                .filter(|node| AxTree::is_trigger(node))
                .filter_map(|node| node.backend)
                .collect::<Vec<_>>();
            match inner.as_slice() {
                [only] => *only,
                _ => return Ok(AriaOutcome::NotADropdown),
            }
        };
        let Some(wanted) = wanted else {
            return Ok(AriaOutcome::NoOption);
        };
        let (_, host_node) = tree.by_backend(host).expect("host found above");
        let editable = host_node.flag("editable");
        let collapsed = host_node
            .property("expanded")
            .and_then(|value| value.get("value"))
            .and_then(Value::as_bool)
            == Some(false);
        let is_list = host_node.role == "listbox";
        let before_shown = self.shown(&tree, host).await;
        let before = tree.option_backends();

        let mut opened = false;
        let mut current = (tree, Vec::new());
        current.1 = current.0.options_for(host, &HashSet::new());
        if !is_list && (collapsed || current.1.is_empty()) {
            self.click(host).await?;
            opened = true;
            current = self.wait_options(host, &before, OPEN_WAIT).await?;
        }
        let mut seen: Vec<String> = Vec::new();
        let remember = |tree: &AxTree, options: &[String], seen: &mut Vec<String>| {
            for id in options {
                let name = tree.nodes[id].name.clone();
                if !name.is_empty() && !seen.contains(&name) {
                    seen.push(name);
                }
            }
        };
        let mut picked = pick_option(&current.0, &current.1, wanted);
        remember(&current.0, &current.1, &mut seen);

        if matches!(picked, Picked::None) && editable {
            self.type_search(host, wanted).await?;
            current = self.wait_options(host, &before, SEARCH_WAIT).await?;
            picked = pick_option(&current.0, &current.1, wanted);
            remember(&current.0, &current.1, &mut seen);
        }

        if matches!(picked, Picked::None) && !current.1.is_empty() {
            if let Some(list) = current.0.list_of(&current.1[0]) {
                let deadline = tokio::time::Instant::now() + LIST_SCROLL_BUDGET;
                let names = |tree: &AxTree, options: &[String]| {
                    options
                        .iter()
                        .map(|id| tree.nodes[id].name.clone())
                        .collect::<Vec<_>>()
                };
                let mut shown_rows = names(&current.0, &current.1);
                for _ in 0..MAX_LIST_SCROLLS {
                    let moved = self.run(list, SCROLL_LIST, false).await == Some(Value::Bool(true));
                    if !moved || tokio::time::Instant::now() >= deadline {
                        break;
                    }
                    // A página redesenha as linhas no evento de scroll: espera-se por elas.
                    let settle = tokio::time::Instant::now() + Duration::from_secs(1);
                    let (tree, options) = loop {
                        tokio::time::sleep(Duration::from_millis(40)).await;
                        let tree = self.tree().await?;
                        let options = tree.options_for(host, &before);
                        if names(&tree, &options) != shown_rows
                            || tokio::time::Instant::now() >= settle
                        {
                            break (tree, options);
                        }
                    };
                    shown_rows = names(&tree, &options);
                    picked = pick_option(&tree, &options, wanted);
                    remember(&tree, &options, &mut seen);
                    current = (tree, options);
                    if !matches!(picked, Picked::None) {
                        break;
                    }
                }
            }
        }

        match picked {
            Picked::One(id) => {
                let node = &current.0.nodes[&id];
                let option = node.backend.expect("options have a DOM node");
                self.choose(option, node.name.clone(), Some((host, before_shown)))
                    .await
            }
            Picked::Several(matches) => {
                if opened {
                    self.close(host, &before).await;
                }
                Ok(AriaOutcome::Ambiguous(matches))
            }
            Picked::None => {
                if opened {
                    self.close(host, &before).await;
                }
                seen.truncate(MAX_LISTED);
                Ok(AriaOutcome::NoMatch {
                    options: seen,
                    opened,
                })
            }
        }
    }
}
