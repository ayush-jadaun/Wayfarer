import test from 'node:test';
import assert from 'node:assert/strict';
import { parseGrid, astar, smooth, inflate, findRoute, isFree, nextInstruction, routeLength } from '../src/nav.js';

// Two rooms joined by a door in a wall at column 5.
const g = parseGrid({
  cellSize: 1, origin: [0, 0], rows: [
    '###########',
    '#....#....#',
    '#....#....#',
    '#.........#',
    '#....#....#',
    '###########',
  ],
});

test('A* goes through the door, never through walls', () => {
  const p = astar(g, [1, 1], [9, 1]);
  assert.ok(p);
  assert.ok(p.some(([x, z]) => x === 5 && z === 3), 'must pass the door cell');
  for (const [x, z] of p) assert.equal(g.blocked[z * g.w + x], 0);
});

test('no corner cutting and smoothing keeps line of sight', () => {
  const s = smooth(g, astar(g, [1, 1], [9, 1]));
  assert.ok(s.length >= 3 && s.length < 8);
});

test('unreachable goal returns null', () => {
  const closed = parseGrid({ cellSize: 1, origin: [0, 0], rows: ['.#.'] });
  assert.equal(astar(closed, [0, 0], [2, 0]), null);
});

test('findRoute falls back to raw grid when inflation closes the door', () => {
  const route = findRoute(g, [1.5, 1.5], [9.5, 1.5], { inflated: inflate(g, 1) });
  assert.ok(route);
  assert.ok(routeLength(route) > 8);
});

test('collision circle', () => {
  assert.ok(isFree(g, 2.5, 2.5, 0.3));
  assert.ok(!isFree(g, 1.1, 2.5, 0.3));
});

test('instructions', () => {
  // facing -z (yaw 0), target straight ahead
  assert.match(nextInstruction([[0, 0], [0, -5]], [0, 0], 0).text, /straight/);
  // target to the +x side is on the right when facing -z
  assert.match(nextInstruction([[0, 0], [5, 0]], [0, 0], 0).text, /right/);
  // straight now, corner ahead turns right (toward +x)
  assert.equal(nextInstruction([[0, 0], [0, -3], [4, -3]], [0, 0], 0).text, 'Go straight 3.0 m, then right');
});
