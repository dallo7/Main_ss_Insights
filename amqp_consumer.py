# --------------------------------------------------------------------------------------------
# amqp_consumer.py - example consumer of Flightradar24 Live Feed via AMQP 1.0
#
# Based on from https://github.com/Azure/azure-event-hubs-python/blob/master/examples/eph.py
#
# Copyright (c) 2013-2021 Flightradar24 AB, all rights reserved
# --------------------------------------------------------------------------------------------


import asyncio
import logging
from dns.asyncresolver import dns
from typing import Union

from azure.eventhub import parse_connection_string
from azure.eventhub.aio import EventHubConsumerClient
from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore

logging.basicConfig(level=logging.INFO)


class AMQPConsumer:
    def __init__(self, connection_string, consumer_group, storage_connection_string,
                 blob_container_name, proxy_host, proxy_port, proxy_user, proxy_pass,
                 dns_check_interval):
        self.connection_string = connection_string
        self.consumer_group = consumer_group
        self.storage_connection_string = storage_connection_string
        self.blob_container_name = blob_container_name
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.proxy_user = proxy_user
        self.proxy_pass = proxy_pass
        self.on_receive_callback = None
        self.dns_check_interval = dns_check_interval
        self.dns_records = None

    def set_callback(self, on_receive_callback):
        self.on_receive_callback = on_receive_callback

    async def on_event(self, partition_context, event):
        partition_id = partition_context.partition_id
        logging.info(f"Received event from Partition id {partition_id}")
        await partition_context.update_checkpoint(event)
        for content in event.body:
            self.on_receive_callback(content)

    async def on_partition_initialize(self, context):
        logging.info(f"Partition {context.partition_id} initialized")

    async def on_partition_close(self, context, reason):
        logging.info(f"Partition {context.partition_id} has closed, reason {reason}")

    async def on_error(self, context, error):
        if context:
            logging.error(f"Partition {context.partition_id} has a partition related error {error}")
        else:
            logging.error(f"Receiving event has a non-partition error {error}")

    async def monitor_dns(self, sleep: int):
        host = parse_connection_string(self.connection_string).fully_qualified_namespace
        while True:
            try:
                answers = await dns.asyncresolver.resolve(host, 'CNAME')
                if not self.dns_records:
                    self.dns_records = answers
                elif self.dns_records.rrset != answers.rrset:
                    self.dns_records = answers
                    logging.info("Detected DNS change, reconnecting")
                    await self.client.close()
                    return

            except Exception as e:
                logging.error(f"Failed to look up CNAME record for {host} with exception {e}")
                self.dns_records = None

            await asyncio.sleep(sleep)

    def consume(self):
        loop = asyncio.get_event_loop()  # Define loop outside try block
        try:
            while True:
                logging.info("Connecting to Azure Event Hub")

                checkpoint_store = None
                if self._is_storage_checkpoint_enabled():
                    checkpoint_store = BlobCheckpointStore.from_connection_string(
                        self.storage_connection_string,
                        self.blob_container_name)

                # Validate connection string before using it
                if not self.connection_string:
                    raise ValueError("Event Hub connection string is empty!")

                self.client = EventHubConsumerClient.from_connection_string(
                    self.connection_string,
                    consumer_group=self.consumer_group,
                    checkpoint_store=checkpoint_store,
                    http_proxy=self._create_proxy_settings()
                )

                tasks = asyncio.gather(
                    self.client.receive(
                        self.on_event,
                        on_error=self.on_error,
                        on_partition_initialize=self.on_partition_initialize,
                        on_partition_close=self.on_partition_close,
                        starting_position="-1",
                    ),
                    self.monitor_dns(self.dns_check_interval)
                )

                loop.run_until_complete(tasks)
                logging.info("Closed connection to Azure Event Hub")

        except ValueError as e:
            logging.error(f"Invalid connection string: {e}")
        except KeyboardInterrupt:
            logging.info("Shutting down gracefully...")
            for task in asyncio.Task.all_tasks():
                task.cancel()
            loop.run_until_complete(asyncio.sleep(0.1))  # Allow tasks to cancel
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        finally:
            if hasattr(self, 'client'):
                loop.run_until_complete(self.client.close())
            loop.close()

    def _create_proxy_settings(self) -> Union[dict]:
        if self.proxy_host:
            return {
                'proxy_hostname': self.proxy_host,
                'proxy_port': self.proxy_port,
                'username': self.proxy_user,
                'password': self.proxy_pass
            }
        return None

    def _is_storage_checkpoint_enabled(self):
        return self.storage_connection_string and self.blob_container_name