"""
Example: Sequential Human Agent Workflow

This example shows orchestrating human agents in a content approval workflow:
1. Content Creator - human creates initial content
2. Technical Reviewer - human reviews technical accuracy
3. Legal Reviewer - human reviews for compliance
4. Manager Approver - human gives final approval
5. Publisher Agent - automated publishing after all approvals

Demonstrates how to handle human input, timeouts, and approval chains.
"""

from runnable import Pipeline, PythonTask, pickled
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os


def create_initial_content(topic: str, creator_name: str) -> Dict[str, Any]:
    """
    Human Agent 1: Content Creator
    Simulates a human creating initial content that needs approval
    """
    print(f"👤 Content Creator ({creator_name}) working on: {topic}")

    # In a real system, this might:
    # - Send a task to a human via web interface
    # - Wait for content submission
    # - Handle timeouts and notifications

    # Simulated content creation
    print("⏳ Waiting for content creator to submit draft...")
    time.sleep(1)  # Simulate human work time

    content_draft = {
        "content_id": f"CONTENT_{int(time.time())}",
        "topic": topic,
        "creator": creator_name,
        "created_at": datetime.now().isoformat(),
        "status": "draft_submitted",
        "content": {
            "title": f"Complete Guide to {topic}",
            "body": f"""
# Introduction to {topic}

This comprehensive guide covers all aspects of {topic}, including best practices,
implementation details, and real-world examples.

## Key Features
- Feature 1: High-performance capabilities
- Feature 2: Scalable architecture
- Feature 3: User-friendly interface

## Implementation
The implementation follows industry standards and includes proper error handling
and security measures.

## Conclusion
This guide provides everything needed to understand and implement {topic}
successfully.
            """.strip(),
            "tags": [topic.lower(), "guide", "tutorial"],
            "estimated_reading_time": "5 minutes"
        },
        "metadata": {
            "word_count": 150,
            "complexity": "intermediate",
            "target_audience": "developers"
        }
    }

    print(f"✅ Content draft submitted: '{content_draft['content']['title']}'")
    return content_draft


def technical_review(content_data: Dict[str, Any], reviewer_name: str) -> Dict[str, Any]:
    """
    Human Agent 2: Technical Reviewer
    Human reviews content for technical accuracy and completeness
    """
    print(f"🔍 Technical Reviewer ({reviewer_name}) reviewing content...")

    content = content_data["content"]
    print(f"📄 Reviewing: {content['title']}")

    # Simulate human review process
    print("⏳ Technical review in progress...")
    time.sleep(1.5)

    # Simulated human review decision
    # In reality, this would be a web form, email, or collaborative tool
    review_result = {
        **content_data,
        "technical_review": {
            "reviewer": reviewer_name,
            "reviewed_at": datetime.now().isoformat(),
            "status": "approved_with_changes",
            "approval": True,
            "comments": [
                "Technical content is accurate and well-structured",
                "Suggest adding code examples in section 2",
                "Minor typo in implementation section - fixed",
                "Overall quality is excellent"
            ],
            "changes_requested": [
                "Add more specific examples in the features section",
                "Include performance benchmarks if available"
            ],
            "technical_score": 8.5,
            "recommendation": "Approve with minor revisions"
        }
    }

    print(f"✅ Technical review completed - Status: {review_result['technical_review']['status']}")
    print(f"📝 Score: {review_result['technical_review']['technical_score']}/10")
    return review_result


def legal_compliance_review(reviewed_content: Dict[str, Any], legal_reviewer: str) -> Dict[str, Any]:
    """
    Human Agent 3: Legal Reviewer
    Human reviews content for legal compliance, copyright, and policy adherence
    """
    print(f"⚖️ Legal Reviewer ({legal_reviewer}) checking compliance...")

    content = reviewed_content["content"]
    print(f"📋 Legal review for: {content['title']}")

    # Simulate legal review process
    print("⏳ Compliance check in progress...")
    time.sleep(1.2)

    # Simulated legal review
    legal_review_result = {
        **reviewed_content,
        "legal_review": {
            "reviewer": legal_reviewer,
            "reviewed_at": datetime.now().isoformat(),
            "status": "approved",
            "compliance_check": True,
            "issues_found": [],
            "copyright_status": "clear",
            "policy_compliance": "compliant",
            "risk_assessment": "low",
            "comments": [
                "No copyright issues identified",
                "Content complies with company publication policy",
                "No sensitive information disclosed",
                "Approved for public release"
            ],
            "legal_score": 9.0,
            "expiry_date": (datetime.now() + timedelta(days=365)).isoformat()
        }
    }

    print(f"✅ Legal review completed - Status: {legal_review_result['legal_review']['status']}")
    print(f"📊 Risk Level: {legal_review_result['legal_review']['risk_assessment']}")
    return legal_review_result


def manager_final_approval(legal_approved_content: Dict[str, Any], manager_name: str) -> Dict[str, Any]:
    """
    Human Agent 4: Manager Approver
    Manager provides final approval for publication
    """
    print(f"👔 Manager ({manager_name}) reviewing for final approval...")

    content = legal_approved_content["content"]
    technical = legal_approved_content["technical_review"]
    legal = legal_approved_content["legal_review"]

    print(f"📈 Final review for: {content['title']}")
    print(f"   Technical Score: {technical['technical_score']}/10")
    print(f"   Legal Status: {legal['status']}")

    # Simulate manager review
    print("⏳ Manager review in progress...")
    time.sleep(1)

    # Manager decision based on previous reviews
    approval_decision = (
        technical["approval"] and
        legal["compliance_check"] and
        technical["technical_score"] >= 7.0 and
        legal["risk_assessment"] in ["low", "medium"]
    )

    final_approval_result = {
        **legal_approved_content,
        "manager_approval": {
            "approver": manager_name,
            "approved_at": datetime.now().isoformat(),
            "status": "approved" if approval_decision else "rejected",
            "final_approval": approval_decision,
            "business_priority": "high",
            "publish_schedule": "immediate",
            "budget_approved": True,
            "comments": [
                "Excellent work by the content team",
                "Technical and legal reviews are thorough",
                "Content aligns with business objectives",
                "Approved for immediate publication"
            ] if approval_decision else [
                "Content needs additional review",
                "Please address technical reviewer comments",
                "Resubmit after revisions"
            ],
            "approval_conditions": [
                "Monitor engagement metrics after publication",
                "Update content if feedback indicates issues"
            ] if approval_decision else []
        }
    }

    status = "APPROVED" if approval_decision else "REJECTED"
    print(f"✅ Manager review completed - Final Status: {status}")
    return final_approval_result


def automated_publisher(approved_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automated Agent: Publisher
    Automatically publishes content after all human approvals
    """
    print("🚀 Automated Publisher processing final approvals...")

    manager_approval = approved_content["manager_approval"]

    if not manager_approval["final_approval"]:
        print("❌ Publishing cancelled - content not approved")
        return {
            **approved_content,
            "publication_result": {
                "status": "cancelled",
                "reason": "Content not approved by management",
                "published": False,
                "published_at": None
            }
        }

    # Simulate automated publishing process
    print("⏳ Publishing content to various channels...")
    time.sleep(0.8)

    content = approved_content["content"]
    publication_result = {
        **approved_content,
        "publication_result": {
            "status": "published",
            "published": True,
            "published_at": datetime.now().isoformat(),
            "channels": [
                {
                    "platform": "company_website",
                    "url": f"https://company.com/guides/{content['title'].lower().replace(' ', '-')}",
                    "status": "live"
                },
                {
                    "platform": "documentation_site",
                    "url": f"https://docs.company.com/{approved_content['content_id']}",
                    "status": "live"
                },
                {
                    "platform": "internal_wiki",
                    "url": f"https://wiki.company.com/content/{approved_content['content_id']}",
                    "status": "live"
                }
            ],
            "analytics_tracking": True,
            "seo_optimized": True,
            "social_media_scheduled": True
        }
    }

    print(f"✅ Content published successfully to {len(publication_result['publication_result']['channels'])} channels")
    return publication_result


def main():
    """Execute the human agent approval workflow"""

    # Workflow configuration
    workflow_config = {
        "content_topic": "Microservices Architecture",
        "team": {
            "creator": "Sarah Wilson",
            "technical_reviewer": "Dr. James Chen",
            "legal_reviewer": "Maria Garcia",
            "manager": "Robert Thompson"
        }
    }

    # Create human agent workflow pipeline
    pipeline = Pipeline([
        PythonTask(
            function=create_initial_content,
            name="content_creator",
            kwargs={
                "topic": workflow_config["content_topic"],
                "creator_name": workflow_config["team"]["creator"]
            },
            returns=[pickled("initial_content")]
        ),

        PythonTask(
            function=technical_review,
            name="technical_reviewer",
            kwargs={
                "content_data": pickled("initial_content"),
                "reviewer_name": workflow_config["team"]["technical_reviewer"]
            },
            returns=[pickled("technical_reviewed")]
        ),

        PythonTask(
            function=legal_compliance_review,
            name="legal_reviewer",
            kwargs={
                "reviewed_content": pickled("technical_reviewed"),
                "legal_reviewer": workflow_config["team"]["legal_reviewer"]
            },
            returns=[pickled("legal_reviewed")]
        ),

        PythonTask(
            function=manager_final_approval,
            name="manager_approver",
            kwargs={
                "legal_approved_content": pickled("legal_reviewed"),
                "manager_name": workflow_config["team"]["manager"]
            },
            returns=[pickled("final_approved")]
        ),

        PythonTask(
            function=automated_publisher,
            name="publisher",
            kwargs={"approved_content": pickled("final_approved")},
            returns=[pickled("publication_result")]
        )
    ])

    print("🚀 Starting Human Agent Approval Workflow")
    print("=" * 60)
    print(f"📋 Topic: {workflow_config['content_topic']}")
    print(f"👥 Team: {len(workflow_config['team'])} human agents + 1 automated agent")
    print("=" * 60)

    # Execute pipeline
    results = pipeline.execute()

    print("\n" + "=" * 60)
    print("📊 WORKFLOW EXECUTION SUMMARY:")
    print("=" * 60)

    final_result = results["publication_result"]

    # Extract key information
    content_info = final_result["content"]
    technical_review = final_result["technical_review"]
    legal_review = final_result["legal_review"]
    manager_approval = final_result["manager_approval"]
    publication = final_result["publication_result"]

    print(f"📄 Content: {content_info['title']}")
    print(f"👤 Creator: {final_result['creator']}")
    print(f"🔍 Technical Score: {technical_review['technical_score']}/10")
    print(f"⚖️ Legal Status: {legal_review['status']}")
    print(f"👔 Manager Decision: {'APPROVED' if manager_approval['final_approval'] else 'REJECTED'}")
    print(f"🚀 Publication: {'SUCCESS' if publication['published'] else 'FAILED'}")

    if publication['published']:
        print(f"🌐 Published Channels: {len(publication['channels'])}")
        for channel in publication['channels']:
            print(f"   - {channel['platform']}: {channel['status']}")

    print(f"⏰ Workflow Completed: {publication.get('published_at', 'N/A')}")


if __name__ == "__main__":
    main()
