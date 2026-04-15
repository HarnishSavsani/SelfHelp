import asyncio
import sys
import os

# Add root to python path to import everything
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_layer import SQLiteDataLayer, seed_default_users

async def main():
    print("Force Seeding Database...")
    dl = SQLiteDataLayer()
    await seed_default_users(dl)
    await dl.close()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
