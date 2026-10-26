---Makes sure adult horses are drawn with this mod's bodies (the ones with the new tail).
---
---The Horse mod uses the stallion body as its animation skeleton, so the game reads
---HorseMod/Horse/Stallion while it loads the main menu's mods. When this mod is only
---enabled on the save, that early copy is the Horse mod's own, the game keeps it, and
---stallions show the old tail. Reloading the bodies once the save is running reads them
---again, now from this mod. If they were already this mod's, it changes nothing.

local BODIES = { "horsemod/horse/stallion", "horsemod/horse/mare" }

local function reloadBodies()
    for _, mesh in ipairs(BODIES) do
        reloadModelsMatching(mesh)
    end
end

-- once when the save starts, and again a few seconds later for horses that load in after
local ticksLeft = 300

local function reloadLater()
    ticksLeft = ticksLeft - 1
    if ticksLeft <= 0 then
        Events.OnTick.Remove(reloadLater)
        reloadBodies()
    end
end

Events.OnGameStart.Add(function()
    reloadBodies()
    Events.OnTick.Add(reloadLater)
end)
