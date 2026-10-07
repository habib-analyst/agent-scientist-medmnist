"""Minimal state-graph engine with LangGraph-equivalent semantics.

A StateGraph holds named node functions ``fn(state) -> dict`` whose returned
dicts are merged into the shared state, plus edges (unconditional) and
conditional edges (a router ``fn(state) -> str`` naming the next node).
``END`` terminates the run. Every transition is recorded in
``state["_trace"]`` for full auditability — every agent decision is traceable.

This is intentionally dependency-free: ``pip install langgraph`` can replace
this module 1:1 later (same node/edge/conditional-edge vocabulary).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

END = "__end__"

NodeFn = Callable[[dict], dict]
RouterFn = Callable[[dict], str]


class StateGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, NodeFn] = {}
        self._edges: Dict[str, str] = {}
        self._cond_edges: Dict[str, RouterFn] = {}
        self._entry: Optional[str] = None

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in self._nodes:
            raise ValueError(f"duplicate node: {name}")
        self._nodes[name] = fn
        return self

    def set_entry(self, name: str) -> "StateGraph":
        if name not in self._nodes:
            raise ValueError(f"unknown node: {name}")
        self._entry = name
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self._edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: RouterFn) -> "StateGraph":
        self._cond_edges[src] = router
        return self

    def _next(self, node: str, state: dict) -> str:
        if node in self._cond_edges:
            nxt = self._cond_edges[node](state)
            if nxt != END and nxt not in self._nodes:
                raise ValueError(f"router from {node} returned unknown node: {nxt}")
            return nxt
        return self._edges.get(node, END)

    def run(self, initial_state: Optional[dict] = None, max_steps: int = 1000) -> dict:
        if self._entry is None:
            raise ValueError("no entry node set")
        state: dict = dict(initial_state or {})
        state["_trace"] = []
        node = self._entry
        steps = 0
        while node != END:
            if steps >= max_steps:
                raise RuntimeError(f"max_steps={max_steps} exceeded; possible infinite loop")
            fn = self._nodes[node]
            update = fn(state) or {}
            if not isinstance(update, dict):
                raise TypeError(f"node {node} must return a dict, got {type(update)}")
            state.update(update)
            nxt = self._next(node, state)
            state["_trace"].append({"step": steps, "node": node, "next": nxt})
            node = nxt
            steps += 1
        state["_steps"] = steps
        return state

    @property
    def nodes(self) -> List[str]:
        return list(self._nodes)
