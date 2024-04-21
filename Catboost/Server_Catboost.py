import flwr as fl
import sys
import numpy as np

strategy=fl.server.strategy.FedAvg()



fl.server.start_server(
        server_address = "127.0.0.1:18080", 
        config=fl.server.ServerConfig(num_rounds=4), strategy=strategy)