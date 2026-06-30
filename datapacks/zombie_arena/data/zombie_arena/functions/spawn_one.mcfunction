# === Spawn one zombie at a random "every other block" cell ===
# Grid: col rx in 0..12 -> X=-247+2*rx ({-247..-223}), row rz in 0..6 -> Z=-134+2*rz ({-134..-122}), Y=4
# Zombie is summoned at the base corner (-247,4,-134) then moved by reliable relative teleports.

# --- random column rx in 0..12 (4 coin flips -> 0..15, then %13) ---
scoreboard players set #rng arena 0
execute if predicate zombie_arena:coin run scoreboard players add #rng arena 1
execute if predicate zombie_arena:coin run scoreboard players add #rng arena 2
execute if predicate zombie_arena:coin run scoreboard players add #rng arena 4
execute if predicate zombie_arena:coin run scoreboard players add #rng arena 8
scoreboard players operation #rx arena = #rng arena
scoreboard players operation #rx arena %= #c13 arena

# --- random row rz in 0..6 (3 coin flips -> 0..7, then %7) ---
scoreboard players set #rng2 arena 0
execute if predicate zombie_arena:coin run scoreboard players add #rng2 arena 1
execute if predicate zombie_arena:coin run scoreboard players add #rng2 arena 2
execute if predicate zombie_arena:coin run scoreboard players add #rng2 arena 4
scoreboard players operation #rz arena = #rng2 arena
scoreboard players operation #rz arena %= #c7 arena

# --- copy to step counters used by the move loops ---
scoreboard players operation #ix arena = #rx arena
scoreboard players operation #iz arena = #rz arena

# --- summon at base corner cell (-247,4,-134), tagged arena_new, per current mode ---
# mode 1 = static: NoAI + NoGravity so it stays put as a target
execute if score #mode arena matches 1 run summon minecraft:zombie -247 4 -134 {Tags:["arena_zombie","arena_new"],IsBaby:0b,Health:1.0f,NoAI:1b,NoGravity:1b,Silent:1b,PersistenceRequired:1b,CanPickUpLoot:0b,Attributes:[{Name:"minecraft:zombie.spawn_reinforcements",Base:0.0d}]}
# mode 2 = natural: normal AI, will path toward the player
execute if score #mode arena matches 2 run summon minecraft:zombie -247 4 -134 {Tags:["arena_zombie","arena_new"],IsBaby:0b,Health:1.0f,PersistenceRequired:1b,CanPickUpLoot:0b,Attributes:[{Name:"minecraft:zombie.spawn_reinforcements",Base:0.0d}]}

# --- move the new zombie into its cell with reliable relative teleports ---
function zombie_arena:move_x
function zombie_arena:move_z

# --- rotate to face the nearest player (matters for static/NoAI zombies) ---
execute as @e[type=zombie,tag=arena_new,limit=1] at @s run tp @s ~ ~ ~ facing entity @p eyes

# --- clear the temp tag ---
tag @e[type=zombie,tag=arena_new] remove arena_new
