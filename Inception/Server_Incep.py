import flwr as fl
import sys
import numpy as np

strategy=fl.server.strategy.FedAvg()

fl.server.start_server(
        server_address = "[::]:16080", 
        config=fl.server.ServerConfig(num_rounds=7), strategy=strategy)