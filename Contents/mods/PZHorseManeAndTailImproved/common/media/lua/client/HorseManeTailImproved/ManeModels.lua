---Gives every mane bone of an adult horse its own mane model.
---
---The Horse mod puts the same model on the five middle mane bones, so the mane could
---only be as long everywhere. With one model per bone it can hang shorter at the poll
---and the withers and longest at the base of the neck, and each piece fits its own
---part of the neck. Foals keep the Horse mod's manes.
---
---The model is set through the mane item's mod data (InventoryItem:setStaticModel).
---Horses do not save their attached items: the Horse mod recreates the manes every
---session, so nothing of this ends up in a save and removing the mod brings the
---original manes back.

local HorseUtils = require("HorseMod/Utils")
local ManeManager = require("HorseMod/attachments/ManeManager")

local MODEL_BY_SLOT = {
    ManeStart = "HorseManeTailImproved.ManeStart",
    ManeMid1 = "HorseManeTailImproved.ManeMid1",
    ManeMid2 = "HorseManeTailImproved.ManeMid2",
    ManeMid3 = "HorseManeTailImproved.ManeMid3",
    ManeMid4 = "HorseManeTailImproved.ManeMid4",
    ManeMid5 = "HorseManeTailImproved.ManeMid5",
    ManeEnd = "HorseManeTailImproved.ManeEnd",
}

local setupMane = ManeManager.setupMane

---Runs right before the Horse mod attaches each mane item to the horse.
ManeManager.setupMane = function(horse, mane, slot)
    local result = setupMane(horse, mane, slot)
    local model = MODEL_BY_SLOT[slot]
    if model and HorseUtils.isAdult(horse) and mane:getStaticModel() ~= model then
        mane:setStaticModel(model)
    end
    return result
end
