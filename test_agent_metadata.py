import os
import json
from agent_metadata import AgentMetadata

def test_agent_metadata():
    """Test the AgentMetadata class functionality"""
    print("\n=== Testing AgentMetadata ===")
    
    # Delete previous test directory if it exists
    import shutil
    if os.path.exists("test_metadata"):
        shutil.rmtree("test_metadata")
    
    # Create agent metadata instance
    agent = AgentMetadata(
        wallet_address="0x1234567890abcdef1234567890abcdef12345678",
        bot_name="TestBot",
        version="0.1.0",
        strategy_id="secure_ensemble",
        metadata_dir="test_metadata"
    )
    
    # Print agent info
    print("\nAgent Info:")
    agent_info = agent.get_agent_info()
    print(f"Agent ID: {agent_info['agent_id']}")
    print(f"Wallet: {agent_info['wallet_address']}")
    print(f"Strategy: {agent_info['strategy']['name']}")
    
    # Log a simple decision
    print("\nLogging test decision...")
    decision = agent.log_trade_decision(
        decision_type="test",
        symbol="BTCUSDT",
        direction="buy",
        confidence=0.85,
        price=50000.0,
        amount=0.1,
        models_used=["TestModel"]
    )
    
    print(f"Decision ID: {decision['decision_id']}")
    
    # Enable IPFS and upload
    print("\nTesting IPFS mock...")
    agent.enable_ipfs_logging(True)
    
    # Create test data for IPFS
    test_data = {
        "test": True,
        "data": "This is test data",
        "number": 12345
    }
    
    # Upload to mock IPFS
    ipfs_hash = agent.upload_to_ipfs(test_data)
    print(f"IPFS Hash: {ipfs_hash}")
    
    # Check directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk("test_metadata"):
        for directory in dirs:
            print(f"  DIR: {os.path.join(root, directory)}")
        for file in files:
            print(f"  FILE: {os.path.join(root, file)}")
    
    # Check if file was created
    ipfs_dir = os.path.join("test_metadata", "ipfs_mock")
    files_in_dir = os.listdir(ipfs_dir) if os.path.exists(ipfs_dir) else []
    
    print(f"\nFiles in IPFS mock directory: {files_in_dir}")
    
    if files_in_dir:
        filepath = os.path.join(ipfs_dir, files_in_dir[0])
        print(f"Found file: {filepath}")
        
        # Read content to verify
        with open(filepath, 'r') as f:
            content = json.load(f)
            print(f"File content: {json.dumps(content, indent=2)}")
            
            if content.get("test") == True and content.get("data") == "This is test data":
                print("PASS: File content is correct")
            else:
                print("FAIL: File content is incorrect!")
    else:
        print("FAIL: No files found in IPFS mock directory!")
    
    # Test stats
    stats = agent.get_stats()
    print(f"\nDecision count: {stats['decision_count']}")
    
    # Test uploading all decisions to IPFS
    print("\nUploading all decisions to IPFS...")
    all_data_hash = agent.upload_to_ipfs()
    print(f"All data IPFS hash: {all_data_hash}")
    
    print("\n=== Test Complete ===")
    return agent

if __name__ == "__main__":
    agent = test_agent_metadata() 