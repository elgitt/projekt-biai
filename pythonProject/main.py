import methods as m
import model as model

dataL = m.read_from_parquet('0kg_L')
dataR = m.read_from_parquet('0kg_R')
dataW = m.read_from_parquet('0kg_W')

m.generate_graph(dataL, dataR, dataW)

model.model_MLP(dataL, dataR, dataW)
