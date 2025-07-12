import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv


class AzureKeyVaultManager:
    """Azure Key Vault integration for secure credential management"""
    
    def __init__(self, vault_url: Optional[str] = None):
        """Initialize Azure Key Vault client
        
        Args:
            vault_url: Azure Key Vault URL (e.g., https://your-vault.vault.azure.net/)
        """
        load_dotenv()
        
        self.vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL")
        self.client = None
        self.available = False
        
        # Try to initialize Key Vault client
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            
            if self.vault_url:
                credential = DefaultAzureCredential()
                self.client = SecretClient(vault_url=self.vault_url, credential=credential)
                self.available = True
                print(f"✅ Azure Key Vault initialized: {self.vault_url}")
            else:
                print("⚠️  Azure Key Vault URL not provided, falling back to environment variables")
                
        except ImportError:
            print("⚠️  Azure Key Vault libraries not installed. Install with: pip install azure-keyvault-secrets azure-identity")
        except Exception as e:
            print(f"⚠️  Failed to initialize Azure Key Vault: {e}")
    
    def get_secret(self, secret_name: str, fallback_env_var: Optional[str] = None) -> Optional[str]:
        """Get secret from Key Vault with environment variable fallback
        
        Args:
            secret_name: Name of the secret in Key Vault
            fallback_env_var: Environment variable name to use as fallback
            
        Returns:
            Secret value or None if not found
        """
        # Try Key Vault first
        if self.available and self.client:
            try:
                secret = self.client.get_secret(secret_name)
                print(f"✅ Retrieved secret '{secret_name}' from Key Vault")
                return secret.value
            except Exception as e:
                print(f"⚠️  Failed to retrieve secret '{secret_name}' from Key Vault: {e}")
        
        # Fallback to environment variable
        if fallback_env_var:
            value = os.getenv(fallback_env_var)
            if value:
                print(f"✅ Retrieved secret '{secret_name}' from environment variable '{fallback_env_var}'")
                return value
            else:
                print(f"❌ Secret '{secret_name}' not found in Key Vault or environment variable '{fallback_env_var}'")
        
        return None
    
    def get_all_secrets(self) -> Dict[str, str]:
        """Get all secrets from Key Vault (for debugging/admin purposes)
        
        Returns:
            Dictionary of secret names and values
        """
        if not self.available or not self.client:
            return {}
        
        secrets = {}
        try:
            secret_properties = self.client.list_properties_of_secrets()
            for secret_property in secret_properties:
                try:
                    secret = self.client.get_secret(secret_property.name)
                    secrets[secret_property.name] = secret.value
                except Exception as e:
                    print(f"⚠️  Failed to retrieve secret '{secret_property.name}': {e}")
        except Exception as e:
            print(f"⚠️  Failed to list secrets: {e}")
        
        return secrets
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set a secret in Key Vault
        
        Args:
            secret_name: Name of the secret
            secret_value: Value of the secret
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available or not self.client:
            print("❌ Key Vault not available for setting secrets")
            return False
        
        try:
            self.client.set_secret(secret_name, secret_value)
            print(f"✅ Secret '{secret_name}' set in Key Vault")
            return True
        except Exception as e:
            print(f"❌ Failed to set secret '{secret_name}': {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Key Vault is available and configured"""
        return self.available
    
    def get_config_for_rag(self) -> Dict[str, Optional[str]]:
        """Get all RAG-related configuration from Key Vault
        
        Returns:
            Dictionary with RAG configuration values
        """
        config = {
            "azure_openai_endpoint": self.get_secret("azure-openai-endpoint", "AZURE_OPENAI_ENDPOINT"),
            "azure_openai_api_key": self.get_secret("azure-openai-api-key", "AZURE_OPENAI_API_KEY"),
            "openai_api_key": self.get_secret("openai-api-key", "GPT_API_KEY"),
            "azure_document_intelligence_endpoint": self.get_secret("azure-di-endpoint", "AZURE_ENDPOINT"),
            "azure_document_intelligence_key": self.get_secret("azure-di-key", "AZURE_KEY")
        }
        
        return config


def get_azure_credentials_with_keyvault() -> Dict[str, Optional[str]]:
    """Convenience function to get Azure credentials with Key Vault fallback
    
    Returns:
        Dictionary with credential values
    """
    vault_manager = AzureKeyVaultManager()
    return vault_manager.get_config_for_rag()


# Example usage for testing
if __name__ == "__main__":
    print("🔐 Testing Azure Key Vault integration...")
    
    # Initialize Key Vault manager
    kv_manager = AzureKeyVaultManager()
    
    # Test getting configuration
    config = kv_manager.get_config_for_rag()
    
    print("\n📋 Configuration retrieved:")
    for key, value in config.items():
        if value:
            print(f"✅ {key}: {'*' * min(len(value), 10)}...")
        else:
            print(f"❌ {key}: Not found")
    
    print(f"\n🔍 Key Vault available: {kv_manager.is_available()}")