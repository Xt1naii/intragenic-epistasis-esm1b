import requests

def get_fasta(uniprot_id: str) -> dict:
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/search?query={uniprot_id}")
    response = response.json()

    try:
        sequence = response.get("results")[0].get("sequence").get("value")
    except:
        sequence = None
    try:
        gene = response.get('results')[0].get('genes')[0].get('geneName').get('value')
    except:
        gene = None
    try:
        length = response.get("results")[0].get("sequence").get("length")
    except:
        length = None

    return {
        'id': uniprot_id,
        'gene': gene,
        'seq': sequence,
        'length': length
        }

if __name__ == '__main__':
    get_fasta()