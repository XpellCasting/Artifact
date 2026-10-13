import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "../../frontend/test-runtime.js";
import { React } from "../../frontend/test-runtime.js";
import { renderToString } from "../../frontend/test-runtime.js";

test("daily activity opens clicked days including gaps, supports keyboard, and respects disabled state", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false }, appType: "custom" });
  try {
    const { default: Activity } = await server.ssrLoadModule("/src/features/history/DailyActivity.jsx");
    const activity = [{ date: "2026-06-06", valid: 20, invalid: 1 }, { date: "2026-06-07", valid: 0, invalid: 0 }];
    const selected = [];
    const props = { activity, selectedDay: "2026-06-06", onSelect: day => selected.push(day), disabled: false };
    const html = renderToString(React.createElement(Activity, props));
    assert.match(html, /sin reportes/);
    assert.match(html, /Día anterior/);
    assert.deepEqual(selected, []);
    function nodes(tree) {
      if (Array.isArray(tree)) return tree.flatMap(nodes);
      if (!tree || typeof tree !== "object") return [];
      return [tree, ...nodes(tree.props?.children)];
    }
    const bars = nodes(Activity(props)).filter(node => node.props?.role === "button");
    bars[1].props.onClick();
    bars[0].props.onKeyDown({ key: "Enter", preventDefault() {} });
    assert.deepEqual(selected, ["2026-06-07", "2026-06-06"]);
    nodes(Activity({ ...props, disabled: true })).filter(node => node.props?.role === "button")[0].props.onClick();
    assert.equal(selected.length, 2);
  } finally { await server.close(); }
});
