// Run with Node.js. Tests the HTML callback without needing WebGL.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
    const source = fs.readFileSync(path.join(__dirname, '..', 'visualize_voxels_3d.py'), 'utf8');
    const callback = source.match(/CAMERA_SYNC_JS = r"""([\s\S]*?)"""/)[1];
    const listeners = {}, changes = [], timers = new Map(), wheelHandlers = {};
    let time = 0, timerId = 0;
    let toggle;
    const camera = x => ({eye: {x, y: 2, z: 3}, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: 0, z: 1}});
    const plot = {
        layout: {scene: {camera: camera(1)}, scene2: {camera: camera(1)}},
        rendered: {scene: camera(1), scene2: camera(1)},
        parentNode: {insertBefore() {}},
        on(event, handler) { listeners[event] = handler; },
    };
    plot._fullLayout = {};
    for (const scene of ['scene','scene2']) {
        plot._fullLayout[scene] = {_scene: {
            getCamera() { return plot.rendered[scene]; },
            glplot: {canvas: {addEventListener(event, fn) { wheelHandlers[scene] = fn; }}},
        }};
    }
    vm.runInNewContext(callback, {
        document: {
            getElementById() { return plot; },
            createTextNode(text) { return text; },
            createElement(tag) {
                const element = {style: {}, append() {}, addEventListener(event, fn) { this[event] = fn; }};
                if (tag === 'input') toggle = element;
                return element;
            },
        },
        Plotly: {relayout(_, change) {
            changes.push(change);
            for (const key of Object.keys(change)) plot.layout[key.split('.')[0]].camera = change[key];
            // Real relayout redraws both scenes from layout, including the
            // source. A live drag may not yet have committed its layout camera.
            for (const scene of ['scene','scene2']) plot.rendered[scene] = plot.layout[scene].camera;
            listeners.plotly_relayout(change); // Simulate Plotly feedback.
            return Promise.resolve();
        }},
        setTimeout(fn, delay) { const id = ++timerId; timers.set(id, {fn, at: time+delay}); return id; },
        clearTimeout(id) { timers.delete(id); },
        Promise,
    });
    async function flush(milliseconds = 300) {
        time += milliseconds;
        for (let n = 0; n < 5; n++) {
            for (const [id, timer] of [...timers]) {
                if (timer.at <= time) { timers.delete(id); timer.fn(); }
            }
            await Promise.resolve();
        }
    }
    function move(scene, value, event = 'plotly_relayout') {
        plot.rendered[scene] = camera(value);
        if (event === 'plotly_relayout') plot.layout[scene].camera = camera(value);
        if (listeners[event]) listeners[event]({[scene + '.camera']: camera(value)});
    }
    // Reproduce Plotly's pre-commit drag camera: syncing now would redraw the
    // active scene from its old layout camera and visibly snap it back.
    move('scene', 3, 'plotly_relayouting');
    await flush();
    assert.equal(changes.length, 0, 'Never relayout a camera while dragging');
    assert.equal(plot.rendered.scene.eye.x, 3, 'Dragging must retain its live camera');
    move('scene', 4);
    await flush();
    assert.equal(changes.length, 1, 'Own relayout must not recurse');
    assert.equal(plot.layout.scene2.camera.eye.x, 4);
    assert.equal(plot.rendered.scene.eye.x, 4, 'Source must not snap back on redraw');
    move('scene2', -5);
    await flush();
    assert.equal(changes.length, 2);
    assert.equal(plot.layout.scene.camera.eye.x, -5, 'Reverse direction must synchronize');
    assert.equal(plot.rendered.scene2.eye.x, -5, 'Reverse source must not snap back');
    toggle.checked = false;
    toggle.change();
    move('scene2', 8);
    await flush();
    assert.equal(changes.length, 2, 'Unchecked toggle must allow independent rotation');
    toggle.checked = true;
    toggle.change();
    await flush();
    assert.equal(plot.layout.scene.camera.eye.x, 8, 'Re-enabling must align with last moved view');
    move('scene', 10);
    move('scene', 11);
    await flush();
    assert.equal(changes.length, 4, 'Rapid updates must coalesce');
    assert.equal(plot.layout.scene2.camera.eye.x, 11);
    listeners.plotly_relayout({'scene.aspectratio.z': 0.5});
    await flush();
    assert.equal(changes.length, 4, 'Height controls must not trigger camera sync');

    function wheel(scene, finalZoom) {
        wheelHandlers[scene](); // DOM capture precedes Plotly camera updates.
        listeners.plotly_relayout({[scene+'.camera']: plot.layout[scene].camera});
        // Renderer updates later; layout/event still contain the old zoom.
        plot.rendered[scene] = camera(finalZoom);
    }
    wheel('scene', 6);
    await flush(100);
    assert.equal(changes.length, 4, 'Do not redraw between scroll ticks');
    wheel('scene', 4);
    await flush(100);
    assert.equal(changes.length, 4, 'Additional ticks must postpone synchronization');
    wheel('scene', 2);
    await flush(249);
    assert.equal(changes.length, 4, 'Wait for the full wheel quiet period');
    await flush(1);
    assert.equal(changes.length, 5);
    assert.equal(plot.rendered.scene.eye.x, 2, 'Keep final rendered zoom, not stale wheel payload');
    assert.equal(plot.rendered.scene2.eye.x, 2, 'Copy final zoom to other view');
    wheelHandlers.scene2();
    plot.rendered.scene2 = camera(9);
    listeners.plotly_relayout({}); // Last tick can have no camera keys.
    await flush();
    assert.equal(plot.rendered.scene.eye.x, 9, 'Empty wheel events must still sync reverse zoom');
    assert.equal(changes.length, 6, 'Programmatic updates must not loop');
    wheel('scene', 7);
    toggle.checked = false;
    toggle.change();
    await flush();
    assert.equal(changes.length, 6, 'Disabling must cancel pending wheel sync');
    console.log('PASS: stale zoom events, scroll bursts, quiet period, reverse zoom, empty events, no feedback and rotation.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
