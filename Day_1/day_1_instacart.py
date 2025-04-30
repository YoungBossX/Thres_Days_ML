import pandas as pd
from sklearn.decomposition import PCA

order_products = pd.read_csv("./Day_1/instacart/order_products__prior.csv")
products = pd.read_csv("./Day_1/instacart/products.csv")
orders = pd.read_csv("./Day_1/instacart/orders.csv")
aisles = pd.read_csv("./Day_1/instacart/aisles.csv")

table_1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
# print("table_1:\n", table_1)
table_2 = pd.merge(table_1, order_products, on=["product_id", "product_id"])
# print("table_2:\n", table_2)
table_3 = pd.merge(table_2, orders, on=["order_id", "order_id"])
# print("table_3:\n", table_3)

table = pd.crosstab(table_3["user_id"], table_3["aisle"])
print("table:\n", table)

data = table[:10000]

transfer = PCA(n_components=0.95)
data_new = transfer.fit_transform(data)

print("data_new:\n", data_new, "\n", data_new.shape)