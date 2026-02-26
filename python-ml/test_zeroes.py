import os
import json
import torch
from collections import defaultdict

def test_graph_density():
    base_dir = "/Users/blank/Desktop/dating-profile/matcher/python-ml/training-data"
    
    with open(os.path.join(base_dir, "users.json")) as f: users = json.load(f)
    with open(os.path.join(base_dir, "events.json")) as f: events = json.load(f)
    with open(os.path.join(base_dir, "spaces.json")) as f: spaces = json.load(f)
    with open(os.path.join(base_dir, "event_attendees.json")) as f: attends = json.load(f)
    with open(os.path.join(base_dir, "members.json")) as f: members = json.load(f)
    
    print(f"Data Loaded: {len(users)} users, {len(events)} events, {len(spaces)} spaces")
    
    # Check 1: Event to User (event->user)
    # The model tries to predict which user attends.
    event_to_users = defaultdict(list)
    for r in attends:
        event_to_users[r["eventId"]].append(r["userId"])
        
    avg_users_per_event = sum(len(u) for u in event_to_users.values()) / max(1, len(event_to_users))
    print(f"\n[1] event->user density:")
    print(f"    Avg users per event: {avg_users_per_event:.2f}")
    print(f"    Random guess chance (10 / 10000 users) = 0.0010 (0.1%)")
    
    # Check 2: Event to Space (event->space)
    event_to_space = {e["id"]: e.get("spaceId") for e in events if e.get("spaceId")}
    spaces_with_events = set(event_to_space.values())
    print(f"\n[2] event->space density:")
    print(f"    Total spaces: {len(spaces)}")
    print(f"    Spaces that actually host events: {len(spaces_with_events)}")
    print(f"    Random guess chance (10 / 700 spaces) = 0.0142 (1.4%)")
    
    # Check 3: Space to User (space->user)
    space_to_users = defaultdict(list)
    for r in members:
        space_to_users[r["spaceId"]].append(r["userId"])
        
    avg_users_per_space = sum(len(u) for u in space_to_users.values()) / max(1, len(space_to_users))
    print(f"\n[3] space->user density:")
    print(f"    Avg users per space: {avg_users_per_space:.2f}")
    print(f"    Random guess chance (10 / 10000 users) = 0.0010 (0.1%)")

if __name__ == '__main__':
    test_graph_density()
