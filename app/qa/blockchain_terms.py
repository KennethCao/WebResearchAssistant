from typing import Dict, Optional

class BlockchainTerms:
    """Blockchain terminology knowledge base"""
    
    terms = {
        "blockchain": {
            "definition": "A decentralized, distributed ledger technology that records transactions across multiple computers securely",
            "related": ["distributed ledger", "consensus", "cryptography"]
        },
        "consensus mechanism": {
            "definition": "Protocol that determines how network participants agree on the validity of transactions",
            "related": ["proof of work", "proof of stake", "byzantine fault tolerance"]
        },
        "smart contract": {
            "definition": "Self-executing contracts with the terms directly written into code",
            "related": ["ethereum", "solidity", "automated execution"]
        },
        "mining": {
            "definition": "Process of validating transactions and adding them to the blockchain through computational work",
            "related": ["proof of work", "hash rate", "block reward"]
        },
        "proof of work": {
            "definition": "Consensus mechanism requiring computational effort to validate transactions and create new blocks",
            "related": ["mining", "hash function", "difficulty"]
        },
        "proof of stake": {
            "definition": "Consensus mechanism where validators stake tokens to participate in block creation",
            "related": ["staking", "validator", "energy efficiency"]
        },
        "hash function": {
            "definition": "Cryptographic function that converts input data into a fixed-size output string",
            "related": ["SHA256", "cryptography", "data integrity"]
        },
        "wallet": {
            "definition": "Software or hardware that stores private keys and manages cryptocurrency",
            "related": ["private key", "public key", "address"]
        },
        "token": {
            "definition": "Digital asset created and managed on an existing blockchain platform",
            "related": ["cryptocurrency", "smart contract", "ERC20"]
        },
        "defi": {
            "definition": "Decentralized finance - financial services using blockchain technology without traditional intermediaries",
            "related": ["lending", "yield farming", "liquidity pool"]
        }
    }
    
    @classmethod
    def get_term_definition(cls, term: str) -> Optional[Dict]:
        """Get term definition and related concepts"""
        term = term.strip().lower()
        
        # Direct match
        if term in cls.terms:
            return {
                "term": term,
                **cls.terms[term]
            }
            
        # Fuzzy match
        for key in cls.terms:
            if term in key or key in term:
                return {
                    "term": key,
                    **cls.terms[key]
                }
                
        return None 