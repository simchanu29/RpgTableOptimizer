#!/usr/bin/env python3

from Optimizer import OptimizerDeterminist
from Models import EventModel

if __name__ == "__main__":
    # Init event
    model = EventModel(max_parallel=4)

    model.from_csv(
        "in_slots.csv",
        "in_activities.csv",
        "in_preferences.csv" 
    )

    if model.clean_preferences():
        optimizer = OptimizerDeterminist()
        event = optimizer.optimize(model)
        
        event.to_csv("out2.csv")
