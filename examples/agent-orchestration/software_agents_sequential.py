"""
Example: Sequential Software Agent Orchestration

This example shows orchestrating automated software agents/services:
1. Data Fetcher Agent - retrieves data from external APIs
2. Data Validator Agent - validates and cleans the data
3. Database Agent - stores processed data
4. Notification Agent - sends alerts and notifications

Each software agent handles a specific system integration or service.
"""

from runnable import Pipeline, PythonTask, pickled
import json
import time
from typing import Dict, List, Any
from datetime import datetime


def fetch_external_data(api_endpoint: str, auth_token: str) -> Dict[str, Any]:
    """
    Software Agent 1: Data Fetcher
    Simulates fetching data from external APIs or services
    """
    print(f"🌐 Data Fetcher Agent connecting to {api_endpoint}")

    # Simulate API call delay
    time.sleep(0.5)

    # Mock external data response
    external_data = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "source": api_endpoint,
        "records": [
            {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "score": 85},
            {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "score": 92},
            {"id": 3, "name": "Carol Davis", "email": "carol@example.com", "score": 78},
            {"id": 4, "name": "David Wilson", "email": "", "score": -5},  # Invalid data
            {"id": 5, "name": "", "email": "eve@example.com", "score": 150}  # Invalid data
        ],
        "total_count": 5,
        "fetch_metadata": {
            "auth_used": bool(auth_token),
            "response_time_ms": 234
        }
    }

    print(f"✅ Fetched {external_data['total_count']} records from external source")
    return external_data


def validate_and_clean_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Software Agent 2: Data Validator
    Validates, cleans, and transforms the raw data
    """
    print("🔍 Data Validator Agent processing raw data...")

    valid_records = []
    invalid_records = []

    for record in raw_data["records"]:
        # Validation rules
        is_valid = (
            record.get("name", "").strip() != "" and
            record.get("email", "").strip() != "" and
            "@" in record.get("email", "") and
            0 <= record.get("score", -1) <= 100
        )

        if is_valid:
            # Clean and normalize valid records
            clean_record = {
                "id": record["id"],
                "name": record["name"].strip().title(),
                "email": record["email"].strip().lower(),
                "score": record["score"],
                "grade": "A" if record["score"] >= 90 else "B" if record["score"] >= 80 else "C"
            }
            valid_records.append(clean_record)
        else:
            invalid_records.append(record)

    validation_result = {
        "validation_status": "completed",
        "timestamp": datetime.now().isoformat(),
        "source_info": {
            "original_count": raw_data["total_count"],
            "source": raw_data["source"]
        },
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "statistics": {
            "valid_count": len(valid_records),
            "invalid_count": len(invalid_records),
            "success_rate": len(valid_records) / raw_data["total_count"] * 100
        }
    }

    print(f"✅ Validated data: {len(valid_records)} valid, {len(invalid_records)} invalid records")
    return validation_result


def store_in_database(validated_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Software Agent 3: Database Agent
    Simulates storing processed data in database systems
    """
    print("💾 Database Agent storing validated records...")

    # Simulate database operations
    time.sleep(0.3)

    stored_records = []
    for record in validated_data["valid_records"]:
        # Simulate database insertion with generated IDs
        db_record = {
            **record,
            "db_id": f"DB_{record['id']:04d}",
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        stored_records.append(db_record)

    # Also log invalid records for review
    error_log = {
        "invalid_count": len(validated_data["invalid_records"]),
        "error_records": validated_data["invalid_records"],
        "logged_at": datetime.now().isoformat()
    }

    storage_result = {
        "storage_status": "success",
        "timestamp": datetime.now().isoformat(),
        "stored_records": stored_records,
        "error_log": error_log,
        "database_info": {
            "table": "user_scores",
            "records_inserted": len(stored_records),
            "connection": "postgresql://localhost:5432/analytics"
        },
        "validation_summary": validated_data["statistics"]
    }

    print(f"✅ Stored {len(stored_records)} records in database")
    return storage_result


def send_notifications(storage_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Software Agent 4: Notification Agent
    Sends notifications about the completed data processing
    """
    print("📧 Notification Agent sending alerts...")

    # Prepare notification content
    stats = storage_data["validation_summary"]
    notifications_sent = []

    # Email notification
    email_notification = {
        "type": "email",
        "recipient": "admin@company.com",
        "subject": f"Data Processing Complete - {stats['valid_count']} records processed",
        "body": f"""
Data processing pipeline completed successfully.

Summary:
- Records processed: {stats['valid_count']}
- Invalid records: {stats['invalid_count']}
- Success rate: {stats['success_rate']:.1f}%
- Database table: {storage_data['database_info']['table']}

Time: {storage_data['timestamp']}
        """.strip(),
        "sent_at": datetime.now().isoformat(),
        "status": "sent"
    }
    notifications_sent.append(email_notification)

    # Slack notification (if there are errors)
    if stats['invalid_count'] > 0:
        slack_notification = {
            "type": "slack",
            "channel": "#data-alerts",
            "message": f"⚠️ Data processing found {stats['invalid_count']} invalid records. Check error log.",
            "sent_at": datetime.now().isoformat(),
            "status": "sent"
        }
        notifications_sent.append(slack_notification)

    # SMS notification (for high error rates)
    if stats['success_rate'] < 80:
        sms_notification = {
            "type": "sms",
            "recipient": "+1234567890",
            "message": f"ALERT: Data processing success rate only {stats['success_rate']:.1f}%. Immediate review needed.",
            "sent_at": datetime.now().isoformat(),
            "status": "sent"
        }
        notifications_sent.append(sms_notification)

    notification_result = {
        "notification_status": "completed",
        "timestamp": datetime.now().isoformat(),
        "notifications_sent": notifications_sent,
        "summary": {
            "total_notifications": len(notifications_sent),
            "notification_types": [n["type"] for n in notifications_sent]
        },
        "processing_summary": {
            "pipeline_success": True,
            "records_processed": stats['valid_count'],
            "error_count": stats['invalid_count']
        }
    }

    print(f"✅ Sent {len(notifications_sent)} notifications: {', '.join(n['type'] for n in notifications_sent)}")
    return notification_result


def main():
    """Execute the sequential software agent pipeline"""

    # Configuration
    config = {
        "api_endpoint": "https://api.example.com/user-scores",
        "auth_token": "Bearer abc123xyz"
    }

    # Create sequential software agent pipeline
    pipeline = Pipeline([
        PythonTask(
            function=fetch_external_data,
            name="data_fetcher_agent",
            kwargs={
                "api_endpoint": config["api_endpoint"],
                "auth_token": config["auth_token"]
            },
            returns=[pickled("raw_data")]
        ),

        PythonTask(
            function=validate_and_clean_data,
            name="data_validator_agent",
            kwargs={"raw_data": pickled("raw_data")},
            returns=[pickled("validated_data")]
        ),

        PythonTask(
            function=store_in_database,
            name="database_agent",
            kwargs={"validated_data": pickled("validated_data")},
            returns=[pickled("storage_result")]
        ),

        PythonTask(
            function=send_notifications,
            name="notification_agent",
            kwargs={"storage_data": pickled("storage_result")},
            returns=[pickled("notification_result")]
        )
    ])

    print("🚀 Starting Sequential Software Agent Pipeline")
    print("=" * 60)

    # Execute pipeline
    results = pipeline.execute()

    print("\n" + "=" * 60)
    print("📊 PIPELINE EXECUTION SUMMARY:")
    print("=" * 60)

    final_result = results["notification_result"]
    print(f"✅ Pipeline Status: {'SUCCESS' if final_result['processing_summary']['pipeline_success'] else 'FAILED'}")
    print(f"📈 Records Processed: {final_result['processing_summary']['records_processed']}")
    print(f"❌ Error Count: {final_result['processing_summary']['error_count']}")
    print(f"📧 Notifications Sent: {final_result['summary']['total_notifications']}")
    print(f"🔔 Notification Types: {', '.join(final_result['summary']['notification_types'])}")
    print(f"⏰ Completed At: {final_result['timestamp']}")


if __name__ == "__main__":
    main()
