import torch
import torch.nn as nn



class CategoricalEmbedder(nn.Module):
    def __init__(self, num_counties, num_product_types,
                 embedding_dim_county, embedding_dim_business, embedding_dim_product, 
                 embedding_dim_month, embedding_dim_weekday, embedding_dim_hour):
        super(CategoricalEmbedder, self).__init__()
        self.embedding_county = nn.Embedding(num_counties, embedding_dim=embedding_dim_county)
        self.embedding_is_business = nn.Embedding(2, embedding_dim=embedding_dim_business)
        self.embedding_product_type = nn.Embedding(num_product_types, embedding_dim=embedding_dim_product)
        self.embedding_month= nn.Embedding(12, embedding_dim=embedding_dim_month)
        self.embedding_weekday = nn.Embedding(7, embedding_dim=embedding_dim_weekday)
        self.embedding_hour = nn.Embedding(24, embedding_dim=embedding_dim_hour)
        
    def forward(self, county, is_business, product_type, month, weekday, hour):
        month = torch.relu(month - 1)
        embedded_county = self.embedding_county(county)
        embedded_business = self.embedding_is_business(is_business)
        embedded_product = self.embedding_product_type(product_type)
        embedded_month = self.embedding_month(month)
        embedded_weekday = self.embedding_weekday(weekday)
        embedded_hour = self.embedding_hour(hour)
        out = torch.cat((embedded_county, embedded_business, embedded_product, embedded_month, embedded_weekday, embedded_hour), dim=1)
        return out
    
    
class Embedder(nn.Module): 
    def __init__(self, num_counties, num_product_types,
                 embedding_dim_county, embedding_dim_business, embedding_dim_product, 
                 embedding_dim_month, embedding_dim_weekday, embedding_dim_hour):
        super(Embedder, self).__init__()
        self.categorical_embedder = CategoricalEmbedder(num_counties=num_counties, num_product_types=num_product_types, 
                                        embedding_dim_county=embedding_dim_county, embedding_dim_business=embedding_dim_business,
                                        embedding_dim_product=embedding_dim_product, embedding_dim_month=embedding_dim_month, 
                                        embedding_dim_weekday=embedding_dim_weekday, embedding_dim_hour=embedding_dim_hour)
    def forward(self, inputs):
        batch_size, series_len, emb_size = inputs.size()
        county_input = inputs[:,:,2].flatten().to(torch.int32)
        business_input = inputs[:,:,3].flatten().to(torch.int32)
        product_input = inputs[:,:,4].flatten().to(torch.int32)
        month_input = inputs[:,:,7].flatten().to(torch.int32)
        hour_input = inputs[:,:,8].flatten().to(torch.int32)
        weekday_input = inputs[:,:,9].flatten().to(torch.int32)

        categorical_emb = self.categorical_embedder(county=county_input, is_business=business_input, product_type=product_input, 
                           month=month_input, weekday=weekday_input, hour=hour_input)

        categorical_emb = categorical_emb.view(batch_size, series_len, -1)
        out = torch.cat((inputs[:,:,:2],inputs[:,:,10:],categorical_emb), dim=-1)
        return out