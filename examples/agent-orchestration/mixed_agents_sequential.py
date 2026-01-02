"""
Example: Mixed Agent Orchestration (AI + Software + Human)

This example shows a comprehensive customer support incident resolution workflow:
1. AI Agent - Initial triage and classification of customer issue
2. Software Agent - Automated system diagnostics and data gathering
3. Human Agent - Expert analysis and decision making
4. AI Agent - Solution generation and recommendations
5. Software Agent - Automated implementation of approved solutions
6. Human Agent - Quality verification and customer communication
7. AI Agent - Documentation and follow-up automation

Demonstrates complex orchestration with multiple agent types working together.
"""

from runnable import Pipeline, PythonTask, pickled
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


def ai_initial_triage(customer_issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI Agent 1: Initial Triage
    AI analyzes customer issue and classifies priority/category
    """
    print("🤖 AI Triage Agent analyzing customer issue...")

    issue_text = customer_issue["description"]
    print(f"📝 Issue: {issue_text[:100]}...")

    # Simulate AI analysis
    time.sleep(0.8)

    # AI classification and analysis
    triage_result = {
        "ticket_id": customer_issue["ticket_id"],
        "customer_info": customer_issue["customer_info"],
        "original_issue": customer_issue,
        "ai_analysis": {
            "category": "performance_issue",
            "subcategory": "database_slowdown",
            "priority": "high",
            "urgency": "medium",
            "complexity": "moderate",
            "estimated_resolution_time": "2-4 hours",
            "confidence_score": 0.87,
            "keywords_extracted": ["slow", "database", "timeout", "queries", "performance"],
            "similar_incidents": [
                {"id": "INC-2023-1245", "resolution": "query_optimization", "similarity": 0.82},
                {"id": "INC-2023-0987", "resolution": "index_rebuild", "similarity": 0.71}
            ],
            "recommended_escalation": False,
            "initial_suggestions": [
                "Check database connection pool",
                "Review recent query performance",
                "Analyze server resource utilization"
            ]
        },
        "ai_timestamp": datetime.now().isoformat()
    }

    priority = triage_result["ai_analysis"]["priority"]
    category = triage_result["ai_analysis"]["category"]
    print(f"✅ AI Triage Complete - Priority: {priority.upper()}, Category: {category}")
    return triage_result


def software_system_diagnostics(triage_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Software Agent 1: System Diagnostics
    Automated system checks and data gathering
    """
    print("⚙️ System Diagnostics Agent running automated checks...")

    category = triage_data["ai_analysis"]["category"]
    print(f"🔍 Running diagnostics for: {category}")

    # Simulate system diagnostic checks
    time.sleep(1.5)

    diagnostics_result = {
        **triage_data,
        "system_diagnostics": {
            "diagnostic_timestamp": datetime.now().isoformat(),
            "system_health": {
                "database_status": "degraded",
                "cpu_usage": 78.5,
                "memory_usage": 82.1,
                "disk_io": "high",
                "network_latency": 45.2,
                "active_connections": 1247,
                "connection_pool_usage": 89.3
            },
            "performance_metrics": {
                "avg_query_time": "2.3s",
                "slow_queries_count": 23,
                "failed_queries_count": 5,
                "cache_hit_ratio": 0.67,
                "deadlock_count": 2
            },
            "error_logs": [
                {"time": "2024-01-15T10:30:15", "level": "ERROR", "message": "Query timeout after 30s"},
                {"time": "2024-01-15T10:28:42", "level": "WARN", "message": "High connection pool usage"},
                {"time": "2024-01-15T10:25:33", "level": "ERROR", "message": "Deadlock detected"}
            ],
            "resource_analysis": {
                "bottleneck_identified": "database_connection_pool",
                "root_cause_probability": 0.78,
                "affected_services": ["user_service", "payment_service", "notification_service"]
            },
            "automated_checks_passed": 12,
            "automated_checks_failed": 3
        }
    }

    status = diagnostics_result["system_diagnostics"]["system_health"]["database_status"]
    bottleneck = diagnostics_result["system_diagnostics"]["resource_analysis"]["bottleneck_identified"]
    print(f"✅ Diagnostics Complete - Status: {status}, Bottleneck: {bottleneck}")
    return diagnostics_result


def human_expert_analysis(diagnostic_data: Dict[str, Any], expert_name: str) -> Dict[str, Any]:
    """
    Human Agent 1: Expert Analysis
    Human expert analyzes complex technical data and makes decisions
    """
    print(f"👨‍💻 Technical Expert ({expert_name}) analyzing diagnostics...")

    ai_analysis = diagnostic_data["ai_analysis"]
    system_data = diagnostic_data["system_diagnostics"]

    print(f"📊 Reviewing: {ai_analysis['category']} - {system_data['system_health']['database_status']}")

    # Simulate human expert analysis
    print("⏳ Expert analysis in progress...")
    time.sleep(2)

    # Human expert decision and analysis
    expert_analysis = {
        **diagnostic_data,
        "expert_analysis": {
            "analyst": expert_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "technical_assessment": {
                "root_cause": "Connection pool exhaustion due to inefficient query patterns",
                "severity": "high",
                "business_impact": "Customer-facing services experiencing 40% slowdown",
                "risk_level": "medium",
                "escalation_needed": False
            },
            "recommended_actions": [
                {
                    "action": "immediate_connection_pool_scaling",
                    "priority": "urgent",
                    "estimated_time": "15 minutes",
                    "risk": "low",
                    "automation_possible": True
                },
                {
                    "action": "query_optimization_deployment",
                    "priority": "high",
                    "estimated_time": "1 hour",
                    "risk": "medium",
                    "automation_possible": True
                },
                {
                    "action": "database_index_rebuild",
                    "priority": "medium",
                    "estimated_time": "2 hours",
                    "risk": "high",
                    "automation_possible": False
                }
            ],
            "approval_decision": "approve_automated_actions",
            "manual_oversight_required": True,
            "confidence_level": 0.92,
            "expert_notes": [
                "Pattern matches known performance issue from Q3",
                "Automated scaling should provide immediate relief",
                "Query optimization patch tested in staging environment",
                "Index rebuild should be scheduled during maintenance window"
            ]
        }
    }

    decision = expert_analysis["expert_analysis"]["approval_decision"]
    confidence = expert_analysis["expert_analysis"]["confidence_level"]
    print(f"✅ Expert Analysis Complete - Decision: {decision}, Confidence: {confidence:.1%}")
    return expert_analysis


def ai_solution_generation(expert_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI Agent 2: Solution Generation
    AI generates detailed implementation steps and code changes
    """
    print("🤖 AI Solution Generator creating implementation plan...")

    expert_analysis = expert_data["expert_analysis"]
    approved_actions = [a for a in expert_analysis["recommended_actions"]
                       if a["automation_possible"] and a["priority"] in ["urgent", "high"]]

    print(f"🎯 Generating solutions for {len(approved_actions)} approved actions")

    # Simulate AI solution generation
    time.sleep(1.2)

    solution_plan = {
        **expert_data,
        "ai_solution": {
            "generation_timestamp": datetime.now().isoformat(),
            "solution_confidence": 0.89,
            "implementation_steps": [
                {
                    "step": 1,
                    "action": "connection_pool_scaling",
                    "description": "Scale database connection pool from 50 to 150 connections",
                    "automation_script": "kubectl scale deployment db-pool --replicas=3",
                    "rollback_plan": "kubectl scale deployment db-pool --replicas=1",
                    "validation_checks": [
                        "connection_pool_usage < 70%",
                        "avg_response_time < 1s",
                        "error_rate < 0.1%"
                    ],
                    "estimated_impact": "60% performance improvement"
                },
                {
                    "step": 2,
                    "action": "query_optimization",
                    "description": "Deploy optimized query patches for slow queries",
                    "automation_script": "docker deploy query-optimizer:v2.1.3",
                    "rollback_plan": "docker deploy query-optimizer:v2.1.2",
                    "validation_checks": [
                        "slow_query_count < 5",
                        "avg_query_time < 0.5s",
                        "cache_hit_ratio > 85%"
                    ],
                    "estimated_impact": "75% query performance improvement"
                }
            ],
            "monitoring_plan": {
                "metrics_to_watch": [
                    "database_response_time",
                    "connection_pool_usage",
                    "error_rate",
                    "customer_satisfaction_score"
                ],
                "alert_thresholds": {
                    "response_time_ms": 1000,
                    "error_rate_percent": 1.0,
                    "pool_usage_percent": 80
                },
                "monitoring_duration": "24 hours"
            },
            "success_criteria": {
                "performance_target": "95% of requests < 1s response time",
                "stability_target": "Error rate < 0.5%",
                "customer_impact": "Zero customer complaints in 4 hours post-fix"
            }
        }
    }

    steps = len(solution_plan["ai_solution"]["implementation_steps"])
    confidence = solution_plan["ai_solution"]["solution_confidence"]
    print(f"✅ Solution Generated - {steps} implementation steps, Confidence: {confidence:.1%}")
    return solution_plan


def software_automated_implementation(solution_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Software Agent 2: Automated Implementation
    Executes approved automated solutions
    """
    print("⚙️ Automated Implementation Agent executing solutions...")

    solution_steps = solution_data["ai_solution"]["implementation_steps"]
    print(f"🚀 Executing {len(solution_steps)} implementation steps")

    # Simulate automated implementation
    implementation_results = []

    for step in solution_steps:
        print(f"   Step {step['step']}: {step['description']}")
        time.sleep(0.8)  # Simulate execution time

        # Simulate execution result
        result = {
            "step_number": step["step"],
            "action": step["action"],
            "status": "success",
            "executed_at": datetime.now().isoformat(),
            "execution_time_seconds": 12.3,
            "output": f"Successfully executed: {step['automation_script']}",
            "validation_results": {
                check: "passed" for check in step["validation_checks"]
            },
            "rollback_available": True
        }
        implementation_results.append(result)
        print(f"   ✅ Step {step['step']} completed successfully")

    # Overall implementation result
    implementation_result = {
        **solution_data,
        "implementation_result": {
            "execution_timestamp": datetime.now().isoformat(),
            "overall_status": "success",
            "steps_executed": len(implementation_results),
            "steps_successful": len([r for r in implementation_results if r["status"] == "success"]),
            "steps_failed": len([r for r in implementation_results if r["status"] == "failed"]),
            "execution_details": implementation_results,
            "system_impact": {
                "downtime_seconds": 0,
                "services_affected": 0,
                "rollback_required": False
            },
            "immediate_metrics": {
                "avg_response_time_ms": 450,
                "connection_pool_usage": 68.2,
                "error_rate_percent": 0.12,
                "customer_complaints": 0
            }
        }
    }

    success_rate = implementation_result["implementation_result"]["steps_successful"] / implementation_result["implementation_result"]["steps_executed"] * 100
    print(f"✅ Implementation Complete - Success Rate: {success_rate:.0f}%")
    return implementation_result


def human_verification_and_communication(implementation_data: Dict[str, Any],
                                       support_agent_name: str) -> Dict[str, Any]:
    """
    Human Agent 2: Quality Verification and Customer Communication
    Human verifies solution effectiveness and communicates with customer
    """
    print(f"👥 Support Agent ({support_agent_name}) verifying and communicating...")

    impl_result = implementation_data["implementation_result"]
    original_issue = implementation_data["original_issue"]

    print(f"📞 Contacting customer: {original_issue['customer_info']['name']}")

    # Simulate human verification and communication
    time.sleep(1.5)

    verification_result = {
        **implementation_data,
        "human_verification": {
            "support_agent": support_agent_name,
            "verification_timestamp": datetime.now().isoformat(),
            "quality_check": {
                "solution_effectiveness": "excellent",
                "customer_impact_resolved": True,
                "performance_improvement_confirmed": True,
                "no_side_effects": True,
                "customer_satisfaction": "high"
            },
            "customer_communication": {
                "contact_method": "phone_call",
                "communication_timestamp": datetime.now().isoformat(),
                "customer_response": "very_satisfied",
                "issue_resolution_confirmed": True,
                "additional_concerns": [],
                "customer_feedback": "Issue completely resolved. System is working perfectly now. Thank you for the quick response!"
            },
            "follow_up_actions": [
                {
                    "action": "schedule_follow_up_call",
                    "timeline": "24_hours",
                    "purpose": "ensure_continued_stability"
                },
                {
                    "action": "add_monitoring_alerts",
                    "timeline": "immediate",
                    "purpose": "prevent_recurrence"
                }
            ],
            "case_closure_recommendation": "approve_closure",
            "agent_notes": [
                "Customer confirmed complete resolution",
                "Performance metrics show significant improvement",
                "No additional issues reported",
                "Excellent collaboration between all team members"
            ]
        }
    }

    satisfaction = verification_result["human_verification"]["customer_communication"]["customer_response"]
    recommendation = verification_result["human_verification"]["case_closure_recommendation"]
    print(f"✅ Verification Complete - Customer: {satisfaction}, Recommendation: {recommendation}")
    return verification_result


def ai_documentation_and_followup(verification_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    AI Agent 3: Documentation and Follow-up
    AI creates comprehensive documentation and schedules follow-up actions
    """
    print("🤖 AI Documentation Agent finalizing case...")

    # Extract key information from the entire workflow
    original_issue = verification_data["original_issue"]
    ai_triage = verification_data["ai_analysis"]
    expert_analysis = verification_data["expert_analysis"]
    implementation = verification_data["implementation_result"]
    verification = verification_data["human_verification"]

    print("📝 Generating comprehensive documentation...")
    time.sleep(1)

    # AI-generated documentation and follow-up
    final_documentation = {
        **verification_data,
        "final_documentation": {
            "documentation_timestamp": datetime.now().isoformat(),
            "case_summary": {
                "ticket_id": original_issue["ticket_id"],
                "customer": original_issue["customer_info"]["name"],
                "issue_category": ai_triage["category"],
                "resolution_time_hours": 2.5,
                "agent_involvement": {
                    "ai_agents": 3,
                    "human_agents": 2,
                    "software_agents": 2
                },
                "customer_satisfaction": verification["customer_communication"]["customer_response"]
            },
            "technical_documentation": {
                "root_cause": expert_analysis["technical_assessment"]["root_cause"],
                "solution_applied": "Connection pool scaling + Query optimization",
                "implementation_steps": len(implementation["execution_details"]),
                "performance_improvement": "75% faster response times",
                "monitoring_enabled": True
            },
            "knowledge_base_update": {
                "new_solution_documented": True,
                "similar_cases_linked": True,
                "troubleshooting_steps_updated": True,
                "automation_scripts_saved": True
            },
            "follow_up_schedule": [
                {
                    "type": "automated_monitoring",
                    "schedule": "continuous",
                    "duration": "7_days",
                    "alerts_configured": True
                },
                {
                    "type": "customer_check_in",
                    "schedule": "24_hours",
                    "assigned_to": verification["support_agent"],
                    "automated_reminder": True
                },
                {
                    "type": "system_health_review",
                    "schedule": "weekly",
                    "assigned_to": "engineering_team",
                    "automated_report": True
                }
            ],
            "case_metrics": {
                "total_resolution_time": "2.5 hours",
                "automation_percentage": 71.4,
                "human_touchpoints": 2,
                "customer_satisfaction_score": 9.5,
                "first_contact_resolution": True
            }
        }
    }

    automation_pct = final_documentation["final_documentation"]["case_metrics"]["automation_percentage"]
    satisfaction_score = final_documentation["final_documentation"]["case_metrics"]["customer_satisfaction_score"]
    print(f"✅ Documentation Complete - Automation: {automation_pct:.1f}%, Satisfaction: {satisfaction_score}/10")
    return final_documentation


def main():
    """Execute the mixed agent orchestration workflow"""

    # Sample customer issue
    customer_issue = {
        "ticket_id": "TICKET-2024-001337",
        "created_at": datetime.now().isoformat(),
        "customer_info": {
            "name": "Jennifer Martinez",
            "company": "TechFlow Solutions",
            "tier": "enterprise",
            "contact": "jennifer.martinez@techflow.com"
        },
        "description": """
        Our application has been experiencing severe performance issues since this morning.
        Database queries are taking 10-15 seconds to complete, causing timeouts across
        our user-facing services. Our customers are complaining about slow response times
        and some are unable to complete transactions. This is impacting our business
        operations significantly. Please help resolve this urgently.
        """.strip(),
        "priority": "urgent",
        "reported_by": "customer_portal"
    }

    # Team configuration
    team_config = {
        "technical_expert": "Alex Rodriguez",
        "support_agent": "Emma Thompson"
    }

    # Create mixed agent orchestration pipeline
    pipeline = Pipeline([
        # AI Agent 1: Initial triage
        PythonTask(
            function=ai_initial_triage,
            name="ai_triage_agent",
            kwargs={"customer_issue": customer_issue},
            returns=[pickled("triage_result")]
        ),

        # Software Agent 1: System diagnostics
        PythonTask(
            function=software_system_diagnostics,
            name="system_diagnostics_agent",
            kwargs={"triage_data": pickled("triage_result")},
            returns=[pickled("diagnostic_result")]
        ),

        # Human Agent 1: Expert analysis
        PythonTask(
            function=human_expert_analysis,
            name="technical_expert",
            kwargs={
                "diagnostic_data": pickled("diagnostic_result"),
                "expert_name": team_config["technical_expert"]
            },
            returns=[pickled("expert_result")]
        ),

        # AI Agent 2: Solution generation
        PythonTask(
            function=ai_solution_generation,
            name="ai_solution_agent",
            kwargs={"expert_data": pickled("expert_result")},
            returns=[pickled("solution_result")]
        ),

        # Software Agent 2: Automated implementation
        PythonTask(
            function=software_automated_implementation,
            name="implementation_agent",
            kwargs={"solution_data": pickled("solution_result")},
            returns=[pickled("implementation_result")]
        ),

        # Human Agent 2: Verification and communication
        PythonTask(
            function=human_verification_and_communication,
            name="support_agent",
            kwargs={
                "implementation_data": pickled("implementation_result"),
                "support_agent_name": team_config["support_agent"]
            },
            returns=[pickled("verification_result")]
        ),

        # AI Agent 3: Documentation and follow-up
        PythonTask(
            function=ai_documentation_and_followup,
            name="ai_documentation_agent",
            kwargs={"verification_data": pickled("verification_result")},
            returns=[pickled("final_result")]
        )
    ])

    print("🚀 Starting Mixed Agent Orchestration Workflow")
    print("=" * 70)
    print(f"🎫 Ticket: {customer_issue['ticket_id']}")
    print(f"👤 Customer: {customer_issue['customer_info']['name']} ({customer_issue['customer_info']['company']})")
    print(f"🔥 Priority: {customer_issue['priority']}")
    print(f"🤖 Agents: 3 AI + 2 Software + 2 Human = 7 Total")
    print("=" * 70)

    # Execute pipeline
    results = pipeline.execute()

    print("\n" + "=" * 70)
    print("📊 MIXED AGENT WORKFLOW SUMMARY:")
    print("=" * 70)

    final_result = results["final_result"]
    case_summary = final_result["final_documentation"]["case_summary"]
    metrics = final_result["final_documentation"]["case_metrics"]

    print(f"🎫 Ticket ID: {case_summary['ticket_id']}")
    print(f"👤 Customer: {case_summary['customer']}")
    print(f"📋 Category: {case_summary['issue_category']}")
    print(f"⏱️ Resolution Time: {case_summary['resolution_time_hours']} hours")
    print(f"🤖 AI Agents Used: {case_summary['agent_involvement']['ai_agents']}")
    print(f"👥 Human Agents: {case_summary['agent_involvement']['human_agents']}")
    print(f"⚙️ Software Agents: {case_summary['agent_involvement']['software_agents']}")
    print(f"🎯 Automation Level: {metrics['automation_percentage']:.1f}%")
    print(f"😊 Customer Satisfaction: {metrics['customer_satisfaction_score']}/10")
    print(f"✅ First Contact Resolution: {'YES' if metrics['first_contact_resolution'] else 'NO'}")
    print(f"📈 Performance Improvement: 75% faster response times")

    print(f"\n🎉 Workflow completed successfully with mixed agent collaboration!")


if __name__ == "__main__":
    main()
