# Standard Operating Procedures for Outages

If `users_db` experiences high latency, immediately failover to the read-replica in the us-east-2 region. Do not restart the Auth Service during a database failover.

If the `order_cache_tier_1` Redis cluster goes down, the Order Service will fall back to querying the primary database directly, which will cause massive CPU spikes. In this event, the lead backend engineer for the dependent service must be paged immediately to implement rate limiting, and the squad leader must be notified to spin up emergency cache nodes.