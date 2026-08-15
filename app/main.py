import os
from crewai import Crew, Task, LLM, Process
from src.engine.agents import MarketAgents

# Mistral Configuration
llm_engine = LLM(
    model="mistral/mistral-large-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.0
)

def launch_free_mission():
    print("\n--- ⚡ UNIVERSAL RESEARCH ENGINE: USER-DRIVEN MODE ---")
    
    # 1. Capture user goal
    mission_goal = input("Enter your Analysis Goal: ") or "Compare AI pricing and performance."
    custom_markers = input("What data markers should I hunt for? (comma-separated): ") or "pricing, context, benchmarks"
    marker_list = [m.strip() for m in custom_markers.split(",")]

    # 2. Collect targets
    targets = []
    print("\nAdd Competitors (Leave name empty to finish):")
    while True:
        name = input("Competitor Name: ")
        if not name:
            break
        url = input(f"URL for {name}: ")
        targets.append({
            "name": name,
            "url": url,
            "keywords": marker_list,
            "user_intent": mission_goal
        })

    if not targets:
        print("No targets provided. Exiting.")
        return

    # 3. Initialize Agents
    agents_factory = MarketAgents()
    analyst = agents_factory.research_analyst(llm=llm_engine)
    checker = agents_factory.fact_checker(llm=llm_engine)
    reporter = agents_factory.reporter(llm=llm_engine)

    # 4. Define Dynamic Tasks for each target
    tasks = []
    for t in targets:
        research_task = Task(
            description=(
                f"User Goal: {t['user_intent']}. "
                f"Current Target: {t['name']}. "
                f"Extract the following markers: {', '.join(t['keywords'])}. "
                f"Use the URL: {t['url']}."
            ),
            expected_output="A structured list of raw data points for the specified markers.",
            agent=analyst
        )

        audit_task = Task(
            description=(
                f"Validate the findings for {t['name']}. Ensure the data points "
                f"for {', '.join(t['keywords'])} are accurate for Jan 2026."
            ),
            expected_output="A verified dataset ready for strategic reporting.",
            agent=checker,
            context=[research_task]
        )

        report_task = Task(
            description=(
                f"Create a final verdict for {t['name']} based on the mission: {t['user_intent']}. "
                f"Synthesize the findings for {', '.join(t['keywords'])} into an executive summary."
            ),
            expected_output="Final Markdown Report.",
            agent=reporter,
            context=[audit_task]
        )

        tasks.append((research_task, audit_task, report_task))

    # 5. Execute with memory disabled
    crew = Crew(
        agents=[analyst, checker, reporter],
        tasks=[task for triplet in tasks for task in triplet],
        process=Process.sequential,
        memory=False,
        verbose=True
    )

    print(f"\n🚀 DEPLOYING CREW ON {len(targets)} USER-DEFINED TARGETS...")

    results = crew.kickoff_for_each(inputs=targets)

    # 6. Save results
    for i, result in enumerate(results):
        target_name = targets[i]['name'].replace(' ', '_')
        filename = f"report_{target_name}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result.raw if hasattr(result, 'raw') else str(result))
        print(f"✅ Mission Complete for {targets[i]['name']}. Saved to {filename}")


if __name__ == "__main__":
    launch_free_mission()
