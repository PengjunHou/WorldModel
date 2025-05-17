## Data Structure

### Car

**Attributes:**
- `carla_vehicle`: CARLA API object representing the vehicle entity in the simulation
- `cluster_id`: identifier of the cluster the vehicle belongs to
- `vid`: dentifier of the vehicle
- `speed`, `position`, `rotation`: current speed, global position, and orientation of the vehicle
- `sensors`: dictionary `{sensor_type: [Sensor...]}` storing attached sensor instances


**Methods:**
- `get_speed()`, `get_position()`, `get_rotation()`: retrieve and update state from the CARLA vehicle
- `get_state()`: aggregate and return the current state
- `get_sensor(sensor_type, sensor_id)`, `get_sensor_data(sensor_type, sensor_id, time_step)`: access and retrieve specific sensor readings
- `join_group(group_id)`, `leave_group()`: join or leave a vehicle cluster
- `control(action)`: execute data upload decision based on RL action
- `receive_data(state)`: accept broadcasted cluster-level data for decision making or fusion
- `upload_data()`: upload data

### CarLeader (subclass of Car)

**Additional Attributes:**
- `members`: list of `Car` instances in this cluster

**Methods:**
- `broadcast_data()`: distribute fused global information to member vehicles
- `receive_data()`: collect feedback or external information from members
- `resource_alloc()`: allocate resource to members of the cluster
- `data_fusion()`: perform multi-source fusion of sensor data and member states, producing a global perception result
- `collect_feedback()`: collect feedback of members
- `control(action)`: override `apply_control_upload` to command the leader vehicle itself

### Cluster

**Attributes:**
- `cluster_id`: unique identifier of the vehicle cluster
- `members`: list of `Car` instances in this cluster
- `leader`: the `CarLeader` instance designated as cluster head
- `leader_id`: unique identifier of the leader vehicle

**Methods:**
- `add_member(member)`, `remove_member(member)`: manage membership changes
- `get_members()`, `get_leader()`: query current members and leader
- `set_leader(leader)`: assign the leader and set its `group_id` to `cluster_id`
- `get_state()`: return aggregated state of the entire cluster (members + leader)
- `get_member_states()`: return state information for member vehicles only

## Interaction Flow

1. **Cluster Formation**
   - A scheduling module divides vehicles into clusters based on proximity or communication range.
   - `Cluster.set_leader()` assigns a `CarLeader`, and members call `join_group()` to join their cluster.

2. **Data Collection and Fusion**
   - Each `Car` calls `get_state()` and `get_sensor_data()` to collect its local information.
   - The leader executes `data_fusion()` to merge its own and members’ data, then calls `broadcast_data()` to share results.

3. **Decision and Execution**
   - Upon receiving fused data, each vehicle constructs input comprising its predicted state and cluster graph.
   - A GNN encoder extracts graph-level features, and an RNN/Transformer predicts its future state.
   - The RL policy (`HybridPolicy`) uses these features to output upload/control actions.
   - Vehicles execute `apply_control_upload(action)` to carry out the RL decisions.

## Algorithm Outline

```text
for each time step t do
  # 1. Local data acquisition
  for each car in all_cars do
    state = car.get_state()
    sensor_readings = car.get_sensor_data(...)
  end for

  # 2. Cluster fusion and broadcast
  for each cluster in clusters do
    leader = cluster.get_leader()
    global_perception = leader.data_fusion()
    leader.broadcast_data()
  end for

  # 3. Graph construction and feature extraction
  graph = build_graph(all_cars, communication_links)
  node_features, edge_index = graph.node_features, graph.edge_index
  graph_embedding = GraphEncoder(node_features, edge_index)

  # 4. Temporal state prediction
  history_seq = collect_historical_states(car)
  predicted_state = StatePredictor(history_seq)

  # 5. RL decision
  observation = { 'session_seq': history_seq,
                  'node_feats': node_features,
                  'edge_index': edge_index }
  actions = RLAgent.predict(observation)  # using HybridPolicy

  # 6. Execute actions
  for each (car, action) in zip(all_cars, actions) do
    car.apply_control_upload(action)
  end for
end for
```

This design covers the key data structures, interaction flow, and algorithmic steps. You can extend it by specifying parameter configurations, communication protocols, and fusion strategies as needed.
