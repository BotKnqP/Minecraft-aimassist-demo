# === Runs once per second ===
# If the mode changed since last cycle, clear existing zombies so the switch takes effect now
execute unless score #mode arena = #lastmode arena run kill @e[type=zombie,tag=arena_zombie]
scoreboard players operation #lastmode arena = #mode arena
# Count currently-alive arena zombies into #count
execute store result score #count arena if entity @e[type=zombie,tag=arena_zombie]
# Keep up to 8 targets alive: while under 8, spawn one per cycle at a random grid cell.
# (Re-count between the two checks so a single 0.5s cycle can add up to 2 -> fast refill.)
execute if score #count arena matches ..7 run function zombie_arena:spawn_one
execute store result score #count arena if entity @e[type=zombie,tag=arena_zombie]
execute if score #count arena matches ..7 run function zombie_arena:spawn_one

# Remove arrows stuck in the ground/wall so they don't pile up (entity lag -> stutter/crash). In-flight
# arrows (not yet stuck) are left alone so real hits still register.
kill @e[type=arrow,nbt={inGround:1b}]
