# === Zombie Arena init (runs on every world load / reload) ===

# scoreboard objective used by the whole system
scoreboard objectives add arena dummy

# math constants (fake-players on the 'arena' objective)
scoreboard players set #c2 arena 2
scoreboard players set #c7 arena 7
scoreboard players set #c13 arena 13
scoreboard players set #cnx arena -247
scoreboard players set #cnz arena -134

# defaults (only set if missing / invalid, so player choices persist across loads)
#   #mode : 1 = static (no AI) , 2 = natural (normal AI)
#   #on   : 1 = running , 0 = paused
execute unless score #mode arena matches 1..2 run scoreboard players set #mode arena 1
execute unless score #on arena matches 0..1 run scoreboard players set #on arena 1
scoreboard players operation #lastmode arena = #mode arena

# reset the per-second timer on load
scoreboard players set #t arena 0

# --- self-install the command-block control panel once ---
# Temporarily force-load the chunk so setblock can run even if the player
# loads in far away. The chunk is released again right after.
forceload add -228 -153
execute unless block -228 4 -153 minecraft:repeating_command_block run function zombie_arena:setup
forceload remove -228 -153
