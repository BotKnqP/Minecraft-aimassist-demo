# === Places the command-block control panel ===
# Line starts at -228 4 -153 and extends NORTH (-Z), one block per control.
# WARNING: this overwrites blocks at -228 4 -153 .. -228 4 -158.

# -153  CLOCK  (repeating, always active) -> drives the 1-second loop
setblock -228 4 -153 minecraft:repeating_command_block[facing=north]{auto:1b,TrackOutput:1b,UpdateLastExecution:1b,CustomName:'{"text":"CLOCK"}',Command:"function zombie_arena:clock"}

# -154  START  (impulse, needs redstone) -> resume spawning
setblock -228 4 -154 minecraft:command_block[facing=north]{auto:0b,TrackOutput:1b,CustomName:'{"text":"START"}',Command:"scoreboard players set #on arena 1"}

# -155  STOP   (impulse, needs redstone) -> pause spawning (keeps existing zombies)
setblock -228 4 -155 minecraft:command_block[facing=north]{auto:0b,TrackOutput:1b,CustomName:'{"text":"STOP"}',Command:"scoreboard players set #on arena 0"}

# -156  MODE STATIC  (impulse) -> new zombies spawn with no AI
setblock -228 4 -156 minecraft:command_block[facing=north]{auto:0b,TrackOutput:1b,CustomName:'{"text":"MODE_STATIC"}',Command:"scoreboard players set #mode arena 1"}

# -157  MODE NATURAL (impulse) -> new zombies spawn with normal AI
setblock -228 4 -157 minecraft:command_block[facing=north]{auto:0b,TrackOutput:1b,CustomName:'{"text":"MODE_NATURAL"}',Command:"scoreboard players set #mode arena 2"}

# -158  CLEAR  (impulse) -> remove all arena zombies
setblock -228 4 -158 minecraft:command_block[facing=north]{auto:0b,TrackOutput:1b,CustomName:'{"text":"CLEAR"}',Command:"kill @e[type=zombie,tag=arena_zombie]"}
