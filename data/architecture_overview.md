# AcmeCorp E-Commerce Architecture Overview

The AcmeCorp platform is built on a microservices architecture. At the edge, we use the Kong API Gateway to route all incoming traffic.

The gateway routes user authentication requests to the Auth Service. The Auth Service relies on a PostgreSQL database named `users_db` to store credentials.

Product browsing is handled by the Catalog Service, which reads from a highly available MongoDB cluster. When a user places an item in their cart, the request goes to the Order Service. The Order Service is critical; it depends on both the Catalog Service for pricing and the Inventory Service to reserve physical stock. 

To handle high traffic, the Order Service caches recent transactions in a Redis cluster named `order_cache_tier_1`.