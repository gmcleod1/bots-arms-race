"""Red team: an autonomous agent that designs and runs a bot farm inside the simulator.

The agent picks its own tactics from the tool surface in `tools.py`. That surface
defines the possibility space (every knob it can turn); it never says what to turn
them to. The agent sees only what a real attacker could: public platform data and
which of its accounts got suspended. It never sees detector scores or signals.
"""
