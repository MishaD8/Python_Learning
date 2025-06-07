import requests
import json
from typing import Dict, Any, Optional

class AlchemyAPIInterface:
    def __init__(self, api_key: str):
        """
        Initialize the Alchemy API interface
        
        Args:
            api_key (str): Your Alchemy API key
        """
        self.api_key = api_key
        self.base_url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
        self.nft_base_url = f"https://eth-mainnet.g.alchemy.com/nft/v3/{api_key}"
        
    def make_request(self, url: str, payload: Dict[str, Any] = None, method: str = "GET") -> Dict[str, Any]:
        """
        Make HTTP request to Alchemy API with proper error handling
        
        Args:
            url (str): API endpoint URL
            payload (dict): Request payload for POST requests
            method (str): HTTP method (GET or POST)
            
        Returns:
            dict: API response or error information
        """
        headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }
        
        try:
            if method.upper() == "POST":
                response = requests.post(url, json=payload, headers=headers, timeout=30)
            else:
                response = requests.get(url, headers=headers, timeout=30)
            
            # Check for successful response (2xx status codes)
            if 200 <= response.status_code < 300:
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "data": response.json()
                }
            else:
                # Handle error responses (4xx, 5xx status codes)
                error_message = "Unknown error"
                if response.status_code == 400:
                    error_message = "Bad Request - Invalid parameters"
                elif response.status_code == 401:
                    error_message = "Unauthorized - Invalid API key"
                elif response.status_code == 404:
                    error_message = "Not Found - Resource does not exist"
                elif response.status_code == 429:
                    error_message = "Rate Limited - Too many requests"
                elif response.status_code >= 500:
                    error_message = "Server Error - Try again later"
                
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": error_message,
                    "details": response.text
                }
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timeout"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Connection error"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def get_nfts_for_owner(self, owner_address: str) -> Dict[str, Any]:
        """Get NFTs owned by a specific address"""
        url = f"{self.nft_base_url}/getNFTsForOwner"
        params = f"?owner={owner_address}&withMetadata=true&pageSize=10"
        return self.make_request(url + params)
    
    def get_nft_metadata(self, contract_address: str, token_id: str) -> Dict[str, Any]:
        """Get metadata for a specific NFT"""
        url = f"{self.nft_base_url}/getNFTMetadata"
        params = f"?contractAddress={contract_address}&tokenId={token_id}"
        return self.make_request(url + params)
    
    def get_nfts_for_collection(self, contract_address: str) -> Dict[str, Any]:
        """Get NFTs from a specific collection/contract"""
        url = f"{self.nft_base_url}/getNFTsForCollection"
        params = f"?contractAddress={contract_address}&withMetadata=true&limit=10"
        return self.make_request(url + params)
    
    def get_eth_balance(self, address: str) -> Dict[str, Any]:
        """Get ETH balance for an address"""
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, "latest"]
        }
        return self.make_request(self.base_url, payload, "POST")
    
    def get_transaction_by_hash(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction details by hash"""
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "eth_getTransactionByHash",
            "params": [tx_hash]
        }
        return self.make_request(self.base_url, payload, "POST")

def extract_nft_info(nft_data: Dict[str, Any]) -> None:
    """Extract and display 5+ pieces of information from NFT response"""
    if not nft_data.get("success"):
        print(f"❌ Error: {nft_data.get('error', 'Unknown error')}")
        return
    
    data = nft_data.get("data", {})
    
    if "ownedNfts" in data:
        # NFTs for owner response
        nfts = data.get("ownedNfts", [])
        total_count = data.get("totalCount", 0)
        
        print(f"\n📊 NFT Collection Summary:")
        print(f"1. Total NFTs owned: {total_count}")
        print(f"2. NFTs in this page: {len(nfts)}")
        
        if nfts:
            first_nft = nfts[0]
            print(f"3. First NFT Contract: {first_nft.get('contract', {}).get('address', 'N/A')}")
            print(f"4. First NFT Token ID: {first_nft.get('tokenId', 'N/A')}")
            print(f"5. First NFT Name: {first_nft.get('name', 'N/A')}")
            
            metadata = first_nft.get("metadata", {})
            if metadata:
                print(f"6. First NFT Description: {metadata.get('description', 'N/A')[:100]}...")
                print(f"7. First NFT Image URL: {metadata.get('image', 'N/A')}")
    
    elif "nfts" in data:
        # NFTs for collection response
        nfts = data.get("nfts", [])
        print(f"\n📊 Collection Summary:")
        print(f"1. NFTs in collection: {len(nfts)}")
        
        if nfts:
            first_nft = nfts[0]
            print(f"2. Contract Address: {first_nft.get('contract', {}).get('address', 'N/A')}")
            print(f"3. Token Standard: {first_nft.get('contract', {}).get('tokenType', 'N/A')}")
            print(f"4. First Token ID: {first_nft.get('tokenId', 'N/A')}")
            print(f"5. First NFT Name: {first_nft.get('name', 'N/A')}")
    
    elif "contract" in data:
        # Single NFT metadata response
        print(f"\n📊 NFT Details:")
        print(f"1. Contract Address: {data.get('contract', {}).get('address', 'N/A')}")
        print(f"2. Token ID: {data.get('tokenId', 'N/A')}")
        print(f"3. Token Standard: {data.get('contract', {}).get('tokenType', 'N/A')}")
        print(f"4. NFT Name: {data.get('name', 'N/A')}")
        print(f"5. NFT Description: {data.get('description', 'N/A')[:100]}...")
        
        metadata = data.get("metadata", {})
        if metadata:
            print(f"6. Image URL: {metadata.get('image', 'N/A')}")
            attributes = metadata.get("attributes", [])
            print(f"7. Number of Attributes: {len(attributes)}")

def extract_transaction_info(tx_data: Dict[str, Any]) -> None:
    """Extract and display transaction information"""
    if not tx_data.get("success"):
        print(f"❌ Error: {tx_data.get('error', 'Unknown error')}")
        return
    
    result = tx_data.get("data", {}).get("result")
    if not result:
        print("❌ Transaction not found")
        return
    
    print(f"\n💰 Transaction Details:")
    print(f"1. Transaction Hash: {result.get('hash', 'N/A')}")
    print(f"2. From Address: {result.get('from', 'N/A')}")
    print(f"3. To Address: {result.get('to', 'N/A')}")
    print(f"4. Value (Wei): {result.get('value', 'N/A')}")
    print(f"5. Gas Price: {result.get('gasPrice', 'N/A')}")
    print(f"6. Gas Used: {result.get('gas', 'N/A')}")
    print(f"7. Block Number: {result.get('blockNumber', 'N/A')}")

def extract_balance_info(balance_data: Dict[str, Any]) -> None:
    """Extract and display balance information"""
    if not balance_data.get("success"):
        print(f"❌ Error: {balance_data.get('error', 'Unknown error')}")
        return
    
    result = balance_data.get("data", {}).get("result")
    if result:
        wei_balance = int(result, 16)
        eth_balance = wei_balance / (10**18)
        
        print(f"\n💰 Balance Information:")
        print(f"1. Balance in Wei: {wei_balance}")
        print(f"2. Balance in ETH: {eth_balance:.6f}")
        print(f"3. Hex Value: {result}")
        print(f"4. Address Type: Ethereum Address")
        print(f"5. Currency: ETH (Ether)")

def main():
    print("🚀 Welcome to Alchemy API Interface!")
    print("=" * 50)
    
    # Get API key from user
    api_key = input("Enter your Alchemy API key: ").strip()
    if not api_key:
        print("❌ API key is required!")
        return
    
    api = AlchemyAPIInterface(api_key)
    
    while True:
        print("\n📋 Available Operations:")
        print("1. Get NFTs for Owner")
        print("2. Get NFT Metadata")
        print("3. Get NFTs for Collection")
        print("4. Get ETH Balance")
        print("5. Get Transaction by Hash")
        print("6. Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == "1":
            address = input("Enter owner address: ").strip()
            if address:
                print("\n🔍 Fetching NFTs for owner...")
                result = api.get_nfts_for_owner(address)
                extract_nft_info(result)
            else:
                print("❌ Address is required!")
        
        elif choice == "2":
            contract = input("Enter contract address: ").strip()
            token_id = input("Enter token ID: ").strip()
            if contract and token_id:
                print("\n🔍 Fetching NFT metadata...")
                result = api.get_nft_metadata(contract, token_id)
                extract_nft_info(result)
            else:
                print("❌ Both contract address and token ID are required!")
        
        elif choice == "3":
            contract = input("Enter collection contract address: ").strip()
            if contract:
                print("\n🔍 Fetching collection NFTs...")
                result = api.get_nfts_for_collection(contract)
                extract_nft_info(result)
            else:
                print("❌ Contract address is required!")
        
        elif choice == "4":
            address = input("Enter Ethereum address: ").strip()
            if address:
                print("\n🔍 Fetching ETH balance...")
                result = api.get_eth_balance(address)
                extract_balance_info(result)
            else:
                print("❌ Address is required!")
        
        elif choice == "5":
            tx_hash = input("Enter transaction hash: ").strip()
            if tx_hash:
                print("\n🔍 Fetching transaction details...")
                result = api.get_transaction_by_hash(tx_hash)
                extract_transaction_info(result)
            else:
                print("❌ Transaction hash is required!")
        
        elif choice == "6":
            print("👋 Thank you for using Alchemy API Interface!")
            break
        
        else:
            print("❌ Invalid choice! Please select 1-6.")
        
        # Ask if user wants to try again after an error
        if choice in ["1", "2", "3", "4", "5"]:
            continue_choice = input("\nWould you like to perform another operation? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("👋 Thank you for using Alchemy API Interface!")
                break

if __name__ == "__main__":
    main()