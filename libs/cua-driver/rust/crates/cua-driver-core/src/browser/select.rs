//! `browser_select`: escolher uma opção num dropdown sem rato nem teclado.
//!
//! O menu de um `<select>` nativo é desenhado pelo browser fora da página
//! (no macOS, um NSMenu): o CDP não o vê nem o opera, e um clique sintético
//! numa `<option>` não escolhe nada. Por isso a opção escolhe-se pela própria
//! página, como o `selectOption` do Playwright. As combobox ARIA abrem-se e
//! escolhem-se com eventos na página, sem levantar a janela.

use std::sync::Arc;

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
use super::tools::{browser_protected_resource_scope, browser_resource_ownership, quad_center};

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

const ARIA_SELECT: &str = r#"async function(wanted) {
    __MATCH__
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const textOf = (n) => String(n.getAttribute('aria-label') || n.innerText || n.textContent || '').replace(/\s+/g, ' ').trim();
    const visible = (n) => {
        const r = n.getBoundingClientRect();
        const s = getComputedStyle(n);
        return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
    };
    const fire = (n) => {
        const r = n.getBoundingClientRect();
        const at = { bubbles: true, cancelable: true, composed: true, view: window, button: 0, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 };
        n.dispatchEvent(new PointerEvent('pointerdown', Object.assign({ pointerType: 'mouse', isPrimary: true, buttons: 1 }, at)));
        n.dispatchEvent(new MouseEvent('mousedown', Object.assign({ buttons: 1 }, at)));
        if (n.focus) n.focus({ preventScroll: true });
        n.dispatchEvent(new PointerEvent('pointerup', Object.assign({ pointerType: 'mouse', isPrimary: true }, at)));
        n.dispatchEvent(new MouseEvent('mouseup', at));
        n.dispatchEvent(new MouseEvent('click', at));
    };
    const roleOf = (n) => String((n.getAttribute && n.getAttribute('role')) || '').toLowerCase();
    if (roleOf(this) === 'option') {
        fire(this);
        await sleep(150);
        return { kind: 'aria', label: textOf(this), verified: this.getAttribute('aria-selected') === 'true' };
    }
    // A ref pode ser o invólucro: a combobox é ela ou a única lá dentro.
    let host = this;
    if (!['combobox', 'listbox', 'button'].includes(roleOf(host)) && host.querySelectorAll) {
        const inner = host.querySelectorAll('[role=combobox],[role=listbox],[aria-haspopup=listbox]');
        if (inner.length === 1) host = inner[0];
    }
    const role = roleOf(host);
    const popup = host.getAttribute('aria-haspopup');
    if (role !== 'combobox' && role !== 'listbox' && popup !== 'listbox' && popup !== 'true') return { kind: 'none' };
    if (wanted == null) return { kind: 'aria', error: 'no_option' };
    const root = host.getRootNode();
    const byId = (id) => (root.getElementById && root.getElementById(id)) || document.getElementById(id);
    const findOptions = () => {
        const ids = [host.getAttribute('aria-controls'), host.getAttribute('aria-owns')].join(' ').split(/\s+/).filter(Boolean);
        const scopes = ids.map(byId).filter(Boolean);
        if (role === 'listbox') scopes.unshift(host);
        let found = [];
        for (const scope of scopes) found.push(...scope.querySelectorAll('[role=option]'));
        if (!found.length) found = Array.from(document.querySelectorAll('[role=option]'));
        return found.filter(visible);
    };
    const waitOptions = async (ms) => {
        let found = findOptions();
        for (let waited = 0; !found.length && waited < ms; waited += 100) { await sleep(100); found = findOptions(); }
        return found;
    };
    const stateOf = () => norm((host.value || '') + ' ' + (host.innerText || host.textContent || ''));
    const before = stateOf();
    let options = findOptions();
    if (!options.length && role !== 'listbox') { fire(host); options = await waitOptions(1500); }
    let picked = options.length ? pick(options, textOf, (o) => o.getAttribute('data-value') || o.id, wanted) : { error: 'no_options' };
    const editable = host.localName === 'input' || host.localName === 'textarea';
    if (!picked.option && editable) {
        // Combobox de pesquisa: só mostra as opções que batem com o que se escreve.
        const proto = host.localName === 'input' ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
        Object.getOwnPropertyDescriptor(proto, 'value').set.call(host, String(wanted || ''));
        host.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, inputType: 'insertText', data: String(wanted || '') }));
        await sleep(200);
        options = await waitOptions(2000);
        if (options.length) picked = pick(options, textOf, (o) => o.getAttribute('data-value') || o.id, wanted);
    }
    if (!picked.option) {
        const listed = { options: options.slice(0, 40).map(textOf) };
        host.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', code: 'Escape', bubbles: true, composed: true }));
        return Object.assign({ kind: 'aria' }, listed, picked);
    }
    const option = picked.option;
    const label = textOf(option);
    if (option.scrollIntoView) option.scrollIntoView({ block: 'nearest' });
    fire(option);
    let verified = false;
    for (let waited = 0; !verified && waited < 1000; waited += 100) {
        await sleep(100);
        const now = stateOf();
        verified = option.getAttribute('aria-selected') === 'true' || (now !== before && now.includes(norm(label)));
    }
    return { kind: 'aria', label, verified };
}"#;

fn native_select_js() -> String {
    NATIVE_SELECT.replace("__MATCH__", MATCH_OPTION)
}

fn aria_select_js() -> String {
    ARIA_SELECT.replace("__MATCH__", MATCH_OPTION)
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
                description: "Choose an option in a dropdown of an exactly-bound tab, by ref, fully in the background. Use it for every dropdown instead of clicking or pressing keys: a native <select> (its menu is drawn by the browser outside the page, so clicks and keys cannot operate it) is set through the page and gets the input and change events a person's choice would fire; a custom ARIA combobox or listbox is opened in the page and the matching role=option is clicked. option is matched against the option labels, then values: exact (ignoring case and extra spaces) first, then a unique partial match. get_browser_state (semantic_v2) lists a native select's options on its ref; no match returns the options there are. The ref may also be the option itself, without option. Refused for heuristic bindings.".into(),
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
                            "description": "Ref of the dropdown (a select, combobox or listbox, or its label or wrapper), or of one of its options, from get_browser_state. Refs are invalidated by navigation and by newer snapshots of the same tab."
                        },
                        "option": {
                            "type": "string",
                            "description": "Label or value of the option to choose, as shown on the page (e.g. \"Portugal\"). Omit only when ref is the option itself."
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

        let mut outcome =
            match call_on_node(conn, cdp, &object_id, &native_select_js(), &option, false).await {
                Ok(outcome) => outcome,
                Err(error) => {
                    return ToolResult::error(format!("selecting through the page failed: {error}"))
                }
            };
        if outcome["kind"] == "none" {
            outcome = match call_on_node(conn, cdp, &object_id, &aria_select_js(), &option, true)
                .await
            {
                Ok(outcome) => outcome,
                Err(error) => {
                    return ToolResult::error(format!("selecting through the page failed: {error}"))
                }
            };
        }
        let kind = outcome["kind"].as_str().unwrap_or("none").to_owned();
        let wanted = option.as_str().unwrap_or("");
        match outcome["error"].as_str() {
            None => {}
            Some("no_option") => {
                return ToolResult::error("option is required unless ref is the option itself")
            }
            Some("no_match") => {
                let listed = quoted_list(outcome.get("options"));
                return unavailable(
                    format!(
                        "no option of this dropdown matches {wanted:?}. Its options{}: {}",
                        match outcome["options_total"].as_u64() {
                            Some(total) if total > 40 => format!(" (first 40 of {total})"),
                            _ => String::new(),
                        },
                        if listed.is_empty() {
                            "none visible"
                        } else {
                            &listed
                        }
                    ),
                    json!({ "kind": kind, "options": outcome["options"], "options_total": outcome["options_total"] }),
                );
            }
            Some("no_options") => {
                return unavailable(
                    "the dropdown showed no role=option elements after it was opened in the page"
                        .into(),
                    json!({ "kind": kind }),
                )
            }
            Some("ambiguous") => {
                return unavailable(
                    format!(
                        "{wanted:?} matches several options: {}. Pass the exact label",
                        quoted_list(outcome.get("matches"))
                    ),
                    json!({ "kind": kind, "matches": outcome["matches"] }),
                )
            }
            Some("disabled") => {
                return unavailable("the dropdown is disabled".into(), json!({ "kind": kind }))
            }
            Some("option_disabled") => {
                return unavailable(
                    format!(
                        "the option {:?} is disabled",
                        outcome["label"].as_str().unwrap_or("")
                    ),
                    json!({ "kind": kind }),
                )
            }
            Some(other) => return ToolResult::error(format!("browser_select failed: {other}")),
        }
        if kind == "none" {
            return unavailable(
                format!("{ext_ref} is not a dropdown: no <select>, combobox or listbox was found at or inside it"),
                json!({ "kind": kind }),
            );
        }
        let label = outcome["label"].as_str().unwrap_or("").to_owned();
        if kind == "native" && outcome["kept"] == false {
            return unavailable(
                format!(
                    "the page put the previous choice back after {label:?} was selected; now showing {:?}",
                    outcome["now"].as_str().unwrap_or("")
                ),
                json!({ "kind": kind, "reverted": true }),
            );
        }
        let verified = kind == "native" || outcome["verified"] == true;
        let text = if kind == "native" {
            format!(
                "selected {label:?} in the dropdown {ext_ref} in {tab_id}, through the page; no native menu was opened"
            )
        } else if verified {
            format!("opened the dropdown {ext_ref} in {tab_id} in the page and chose {label:?}")
        } else {
            format!(
                "opened the dropdown {ext_ref} in {tab_id} in the page and clicked {label:?}, but the page did not show it as chosen. Refresh page state and verify"
            )
        };
        ToolResult::text(text).with_structured(json!({
            "status": "ok",
            "target_id": target_id,
            "tab_id": tab_id,
            "ref": ext_ref,
            "frame": entry.frame.kind.as_str(),
            "route": if kind == "native" { "native_select" } else { "aria_listbox" },
            "effect": if verified { "verified" } else { "unverifiable" },
            "label": label,
            "value": outcome.get("value").cloned().unwrap_or(Value::Null),
            "changed": outcome.get("changed").cloned().unwrap_or(Value::Null),
            "native_menu_shown": false,
        }))
    }
}
