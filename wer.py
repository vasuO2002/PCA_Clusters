import requests
import webbrowser

endpoint_url = "http://127.0.0.1:8000/cluster_product_ids/"


params = {
    "num_clusters": 3,
    "pca_components": 3,
    "distance_metric": "cosine"
}


response = requests.get(endpoint_url, params=params)


if response.status_code == 200:
  
    html_content = response.text
    
    with open("cluster_plot.html", "w") as file:
        file.write(html_content)
    print("HTML content saved to 'cluster_plot.html'")
else:
    print(f"Error: {response.status_code} - {response.reason}")
webbrowser.open("cluster_plot.html")