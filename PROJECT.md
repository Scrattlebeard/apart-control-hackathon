# Beyond Empirical Monitoring: Structural Guarantees for AI Control

## CaMeL + GOAP Architectural Synthesis — Reference Document

_Compiled from conversations between Daniel and Zephyr, December 2025 – March 2026. This document externalizes approximately three months of thinking._

---

## 1. Core Thesis

**Control by construction, not by observation.**

Current AI control protocols rely almost entirely on empirical monitoring — trusted models scoring untrusted outputs, with red-teaming to calibrate thresholds. These guarantees are _empirical_: they hold under tested conditions and degrade under distribution shift. The field should prioritize _structural_ and _provable_ guarantees wherever possible, reserving empirical monitoring only for domains where structural approaches are genuinely impossible.

This isn't hypothetical — adjacent fields (prompt injection defense, agent orchestration) have already developed techniques that provide provable guarantees. The control community hasn't absorbed these lessons.

Instead of watching an untrusted model and trying to catch it misbehaving, you constrain the architecture so misbehavior either _cannot express itself_ or _cannot reach critical components_. Two independent research efforts identified complementary mechanisms:

- **GOAP** (via Embabel) constrains the **plan space** — what sequence of actions is possible
- **CaMeL** (via Google DeepMind) constrains the **data flow** — what information reaches which component

Combined, the architecture asks _"could the agent have done something bad given these structural constraints?"_ rather than _"did the agent do something bad?"_

A monitor's question is retrospective and probabilistic. The construction-based question is prospective and (within its domain) deterministic.

---

## 2. The Problem with Empirical Guarantees

The current approach: red-team the protocol, measure attack success rate (ASR), iterate.

Greenblatt et al. (2024) → Ctrl-Z (Bhatt et al.) → BigCodeBench evaluations → Factor(U,T) → all empirical. The guarantee is: "we tested N attack strategies and ASR was below threshold X." This is security-by-evaluation, not security-by-design.

**The degradation problem.** Guarantees are conditioned on: the attack distribution matching what was tested, the trusted monitor's capability gap remaining stable, the task distribution not shifting, and the untrusted model not finding novel attack vectors. These conditions fail precisely when stakes are highest — when models get more capable, when deployment conditions shift, when the threat model evolves.

**Concrete evidence of degradation:**

- **Factor(D,U) (December 2025)** demonstrates that monitoring _plans_ (abstract task decompositions) yields AUROC 0.52 — effectively chance. This isn't a hypothetical distribution-shift argument; abstraction _already_ defeats monitoring in their own evaluation. Plans are abstract. Natural language task descriptions are abstract. The monitor cannot distinguish malice from competence at that level.

- **Schaeffer et al. (February 2026)** shows safety dropping from 99% to 59% when an adversarial model strategically _chooses when to attack_ based on monitor vulnerability. This isn't marginal degradation — it's a regime change. Empirical monitoring degrades under adaptive adversaries, not just distribution shift. Safety is far more sensitive to false positive rate than true positive rate.

These are not edge cases. They are results from _within_ the control literature demonstrating that the primary safety mechanism is brittle in exactly the ways that matter.

**The SQL injection analogy.** Web security went through this exact arc. Detection-based approaches (WAFs, pattern matching) → structural approaches (prepared statements, parameterized queries). The latter won because they make injection _structurally impossible_ regardless of attacker creativity. Current AI control monitoring is the equivalent of "scan every query for suspicious characters." It works until it doesn't, and the failure mode is silent.

---

## 3. The Two Source Architectures

### 3.1 CaMeL — Capability-Mediated Language Model Execution

**Source**: Debenedetti et al., "Defeating Prompt Injections by Design" (Google DeepMind, March 2025). Follow-up: Tallam & Miller, "Operationalizing CaMeL" (2025).

**Core idea**: Prompt injection is a confused deputy problem. Solve it architecturally, not through detection.

**Mechanism**: Dual-LLM pattern with capability tracking:

- **Privileged LLM (P-LLM)**: Sees only the trusted user query. Generates execution plans as code in a restricted Python dialect. Never exposed to untrusted data.
- **Quarantined LLM (Q-LLM)**: Processes untrusted data (emails, web pages, tool outputs). Stripped of all tool-calling capabilities.
- **Custom Python interpreter**: Maintains a dependency graph tracking data provenance. Every value carries capability metadata indicating its origin and flow restrictions.

**Security enforcement**: At each tool invocation, the interpreter checks whether input data satisfies the tool's capability requirements. If `recipient` was extracted from an untrusted document and `send_email` requires a trusted recipient, execution blocks or prompts for user confirmation.

**Results**: On AgentDojo benchmark, CaMeL achieves 67-77% task completion with near-100% attack prevention — a ~7-17% utility cost for eliminating prompt injection as a threat class.

**Key limitation noted by Tallam & Miller**: Current guarantees are empirical rather than formally verified. They recommend mechanized proofs in Coq/Isabelle for noninterference properties.

**Relationship to Simon Willison's dual-LLM proposal (2023)**: CaMeL advances beyond Willison's original concept by adding the capability tracking / taint propagation layer. The dual-LLM split alone is insufficient without data provenance enforcement.

### 3.2 Embabel / GOAP — Deterministic Planning over Typed Action Sets

**Source**: Embabel framework (Rod Johnson, Spring Framework founder). Uses Goal-Oriented Action Planning with A\* search.

**Core idea**: The planning layer should be deterministic and LLM-free. LLMs may execute bounded tasks _within_ actions, but cannot influence which actions exist or what sequence the planner selects.

**Mechanism**: A\* search over a graph where nodes are world states and edges are typed actions.

- Actions are atoms defined at compile time with explicit type signatures
- Input types become preconditions (action can only execute when required typed object exists on blackboard)
- Output types become postconditions
- The blackboard stores immutable, typed objects

**Embabel's distinctive variant**: Rather than manually specifying preconditions/effects symbolically, Embabel infers them from Kotlin/Java type signatures. `StarPerson → Horoscope` means "requires StarPerson, produces Horoscope" without additional annotation.

**The hard boundary**:

| Layer            | Properties                                             | Can LLM Influence?                        |
| ---------------- | ------------------------------------------------------ | ----------------------------------------- |
| GOAP Planner     | Deterministic, enumerated action space, explainable    | No — same inputs always produce same plan |
| Action Execution | May use LLMs, bounded by type signatures, scoped tools | Yes, but only within typed constraints    |
| Blackboard State | Immutable objects, typed storage                       | Only through validated action outputs     |

**Key safety property**: The LLM cannot create new actions or expand the action space at runtime. It can only succeed or fail at producing outputs matching declared types, triggering deterministic replanning on failure.

---

## 4. Why These Compose Naturally

Both systems enforce separation between trusted orchestration and untrusted execution — they draw the boundary differently. CaMeL focuses on **data provenance** (where did this value come from?), GOAP focuses on **action enumeration** (what can the system do?). These concerns are orthogonal.

### 4.1 Natural Intersection Points

1. **Action boundaries as capability checkpoints**: Each `@Action` invocation is exactly where CaMeL's policy enforcement should occur. Embabel already requires explicit `toolGroups` declarations — these map directly to capability sets.

2. **Type signatures as capability requirements**: Embabel's typed preconditions (`StarPerson → Horoscope`) can encode capability requirements. An action requiring a `TrustedEmail` type would structurally prevent it from operating on tainted data.

3. **Blackboard as capability store**: The blackboard's typed storage naturally supports capability metadata. Objects carry provenance labels that propagate through the type system.

4. **Replanning as capability revocation**: When an action fails or produces suspicious output, Embabel's replanning mechanism provides a natural point for capability adjustment or escalation to user confirmation.

### 4.2 Combined Computation Flow

```
User Query → [Trusted Zone: GOAP Planner selects action sequence]
                    ↓
          [Action Boundary: capability check]
                    ↓
         [Bounded Execution: LLM within action]
                    ↓
          [Data Boundary: taint propagation]
                    ↓
         [Blackboard: typed, immutable storage]
                    ↓
         [Next Action: capabilities checked again]
```

### 4.3 Proposed Three-Layer Control System

**Layer 1 — Deterministic Plan Generation (GOAP)**:

- All possible actions enumerated at compile time
- A\* planner operates on type availability + boolean conditions
- No LLM involvement in plan selection
- Plans are explainable sequences of named actions

**Layer 2 — Capability-Mediated Execution (CaMeL)**:

- Each action boundary triggers policy evaluation
- Input capabilities checked against action's declared requirements
- Q-LLM processes untrusted content within actions
- Output capabilities computed via taint propagation

**Layer 3 — Typed State Management (Blackboard + IFC)**:

- All data carries provenance labels (trusted/untrusted/mixed)
- Type system encodes flow restrictions
- Immutable storage prevents retrospective tainting
- Cross-action communication only through validated typed objects

### 4.4 Combined Protocol Pseudocode

```python
def execute_action(action: Action, inputs: Dict[str, TypedValue]) -> TypedValue:
    # GOAP constraint: action must be in enumerated set
    assert action in REGISTERED_ACTIONS

    # Capability constraint: inputs must satisfy declared capabilities
    for param, required_cap in action.capability_requirements.items():
        actual_cap = inputs[param].capability_label
        if not satisfies(actual_cap, required_cap):
            return request_user_confirmation(action, inputs)

    # Bounded execution: LLM within action constraints
    result = action.execute_with_quarantined_llm(inputs)

    # Type constraint: output must match declared type
    assert isinstance(result, action.output_type)

    # Taint propagation: output inherits from inputs
    result.capability_label = join_labels(
        [i.capability_label for i in inputs.values()]
    )

    return result
```

---

## 5. Three Tiers of Control Guarantees

### Tier 1: Structural guarantees (provable)

Where you can impose architectural constraints that make entire attack classes impossible.

**CaMeL — Control/data flow separation:**

- Untrusted data _cannot_ influence control flow — architectural invariant, not detection
- 77% task completion with provable security vs 84% undefended — small capability tax for categorical safety improvement
- You don't detect prompt injections, you make them structurally inert

**GOAP-based orchestration — Planning/execution separation:**

- LLM executes within sandboxed action steps but cannot choose the next action
- Planning is deterministic A\* over developer-defined action space with typed preconditions/effects
- Deterministic, explainable, auditable plans
- Each action gets only the tools relevant to that step — principle of least privilege

**Factor(D,U)'s accidental contribution:**

- The trusted decomposer constraining the untrusted implementer's attack surface is Tier 1 structural work — even though the authors frame it as a monitoring improvement
- Their own language: "D imposes structural prevention: by generating only trusted child tasks, it severely constrains the attack surface available to the untrusted implementer"
- The monitor on implementations is mop-up after the structure has already done the heavy lifting

### Tier 2: Formal verification of bounded domains

For domains where full structural guarantees are impossible but we can do much better than "ask a weaker model."

**Code generation — the hard case:**

- Writing code is inherently uncertain (halting problem) — can't fully verify arbitrary programs
- But current monitoring is absurdly weak: "ask GPT-4 if this bash command looks suspicious"
- The monitor has no formal relationship to the property it's checking

**Mutation testing as strong monitoring:**

- Instead of "does this look suspicious?" → "does this code do what it claims to do?"
- Generate semantic mutations of the submitted code
- If the test suite doesn't distinguish the original from mutations, the tests are inadequate
- If the tests DO distinguish, you have evidence the code's behavior is well-specified
- The monitor isn't pattern-matching on suspiciousness — it's verifying behavioral properties

**Other formal-ish approaches:**

- Property-based testing (QuickCheck-style) for code outputs
- Sandboxed execution with resource constraints (already partially done)
- Static analysis for known-dangerous patterns (syscalls, network access, file operations)
- Information flow analysis and taint tracking

### Tier 3: Empirical monitoring as residual

Only for genuinely irreducible cases where neither structural constraints nor formal methods apply.

- Text-to-text tasks where output is inherently subjective/unverifiable
- Novel tool use in open-ended environments
- Cases where the capability gap between trusted and untrusted is the only lever

Even here, improvements over vanilla trusted monitoring:

- Ensemble execution with diff analysis
- Trust accumulation protocols — graduated trust based on track record
- Collaborative repair — extract value from suspicious actions rather than pure rejection

---

## 6. The Meta-Argument: Defense in Depth with the Right Depth Ordering

**Current literature**: monitoring (Tier 3) is the _primary_ defense, with some Tier 2 (sandboxing).

**Proposed**: invert the priority.

1. First, impose structural constraints that eliminate attack classes entirely (Tier 1)
2. Then, apply formal verification where possible for the residual (Tier 2)
3. Finally, use empirical monitoring only for what remains (Tier 3)

This mirrors how mature security engineering works. You don't protect a database primarily with anomaly detection — you primarily use access control, type systems, and parameterized queries. Detection is the _last_ layer, not the first.

The field's over-indexing on monitors because the research infrastructure (ControlArena included) assumes observation-based control as the paradigm.

---

## 7. The Boundary: Where Construction-Based Control Works and Breaks

### 7.1 The Boundary Marker

"It definitely doesn't work for using Claude Code to vibe-code." — This isn't a throwaway joke; it's the actual boundary.

Vibe-coding is maximally open-ended: the action space is unbounded, the "correct" output is defined by vibes, and the whole point is novel composition. You can't GOAP-plan "write me a game that feels like Balatro but with dice" because you can't enumerate the action space in advance.

### 7.2 Where GOAP Works

| Domain                                                                     | Why It Works                                            |
| -------------------------------------------------------------------------- | ------------------------------------------------------- |
| Structured workflows (document processing, ETL, approvals)                 | Actions are pipeline stages with clear handoffs         |
| Customer service with bounded responses (refund, escalate, template reply) | Finite action menu, clear success states                |
| Compliance/audit checklists                                                | Run through enumerated checks, flag violations          |
| Research assistance (search → retrieve → summarize → cite)                 | Actions are information transformations, not generation |
| Form processing / data extraction                                          | Extract typed fields, validate against schema           |
| Infrastructure provisioning / sysadmin                                     | Enumerated resource operations, observable effects      |

**The pattern**: Tasks where the _sequencing_ requires intelligence but individual actions are categorical operations on typed data.

### 7.3 Where GOAP Breaks

- **Open-ended coding** — Action space is infinite (every possible edit)
- **Creative writing** — The output IS the task, not a means to typed state
- **Exploratory analysis** — Goals emerge from data, can't be pre-specified
- **Conversation** — No terminal goal state
- **Novel problem-solving** — Can't enumerate solution approaches in advance

### 7.4 The Uncomfortable Implication

Many high-stakes agentic tasks — autonomous coding, research, strategic planning — are exactly the tasks where GOAP provides no traction. The tasks that most _need_ control guarantees are the tasks least amenable to enumerated action spaces.

Schaeffer et al.'s 99% → 59% safety degradation under attack selection lands hardest here. For tasks in the constructible set, strategic attack timing is irrelevant — the constraint applies uniformly. For tasks _outside_ the constructible set, where monitoring is the primary defense, the attacker can exploit exactly this sensitivity.

### 7.5 The Pitch

**Not all agent tasks need open-ended reasoning, and for the ones that don't, we're paying a massive safety tax by using monitor-based control when construction-based control would give us stronger guarantees at lower cost.**

---

## 8. Sysadmin as the Tractable Demonstration Domain

Sysadmin/bash is much more tractable than coding.

### 8.1 Why Sysadmin Works

- **Atomic execution**: Bash commands run individually and sequentially. Unlike compiled code that executes as a unit, you get natural checkpoints after every command.
- **Observable effects**: You can actually verify postconditions (`stat`, `systemctl status`, `netstat`, `dpkg -l`). Declared effects can be _checked_, not just trusted.
- **Unix capabilities are a thing**: The OS already has a capability model. File permissions, sudo, network namespaces, seccomp, SELinux. CaMeL-style capability tracking maps almost directly onto real enforcement mechanisms.
- **Finite command vocabulary**: The set of sysadmin-relevant commands is large but enumerable. Each has relatively well-understood preconditions/effects.

### 8.2 Sketch of Sysadmin Action Primitives

```
Primitive              Capabilities          Precondition           Effect
───────────────────────────────────────────────────────────────────────────
install_pkg(name)      sudo, network         —                      pkg installed
write_config(svc,k,v)  sudo, fs:/etc/svc/    pkg(svc) installed     config updated
restart_svc(name)      sudo                  pkg(name) installed    svc running
open_port(p)           sudo, firewall        —                      port accessible
copy_file(src,dst)     fs:src, fs:dst        file(src) exists       file(dst) exists
```

### 8.3 The T/U Schema Generation Architecture

A promising extension: use the trusted/untrusted model split from AI Control at the _action schema_ level.

```
T (trusted, weaker):  Generates action schemas, parameter constraints, effect verifiers
U (untrusted, capable): Proposes actions within T's constraints, executes tasks
```

**What T produces for a given task** (e.g., "Configure nginx reverse proxy for app on port 3000"):

```python
allowed_actions = {
    "write_file": {
        "paths": ["/etc/nginx/sites-available/*", "/etc/nginx/sites-enabled/*"],
        "content_schema": nginx_config_grammar,
    },
    "symlink": {
        "source": "/etc/nginx/sites-available/*",
        "target": "/etc/nginx/sites-enabled/*",
    },
    "restart_service": { "services": ["nginx"] },
    "test_config": { "command": "nginx -t" },
}

effect_verifiers = {
    "write_file": lambda path, content: os.path.exists(path) and validate(content),
    "restart_service": lambda svc: subprocess.run(
        ["systemctl", "is-active", svc]
    ).returncode == 0,
}
```

**Why this might work**:

1. Schema generation is easier than task execution — T doesn't need to know _how_ to configure nginx, just what files nginx configs live in and roughly what they look like. Lower capability requirement.
2. T's output is auditable — A human can review the schema before U executes.
3. Conservative failure mode — If T is uncertain, it generates restrictive schemas. U fails, escalates to human. Safety preserved at cost of utility.
4. Separation of concerns — T reasons about "what could go wrong," U reasons about "how to accomplish the goal."

**The circularity problem**: If T is compromised, T generates permissive schemas, U exploits them. Mitigations: T is weaker (less scheming risk), T's schemas require human approval, multiple independent Ts with consensus, T operates from heavily constrained meta-prompts.

**Connection to Schaeffer et al.**: In the BashBench-style setting, their result implies that an untrusted model could learn to time its attacks for moments of low monitor sensitivity. The T/U schema architecture addresses this differently: if the trusted model has pre-constrained which commands and arguments are valid for a given task, the untrusted model's strategic timing is irrelevant — the constraint applies uniformly across all steps.

### 8.4 Where Sysadmin Still Leaks

- **Arguments**: `write_config(nginx, include, "/etc/shadow")` uses valid primitives but does something awful. Requires schema validation, sandbox execution, or argument provenance tracking.
- **Command injection patterns**: `curl | bash` and similar. Must prohibit piping to execution contexts entirely.
- **The deeper question**: Is schema generation _actually_ easier than the task itself? For sysadmin, probably yes. May not generalize.

---

## 9. What Can and Cannot Be Guaranteed

### 9.1 Structurally Enforced (Deterministic)

- **Bounded action space**: Only compile-time enumerated actions can execute. No amount of LLM manipulation can create actions that weren't defined.
- **Type-enforced preconditions**: Actions cannot execute without required typed inputs. The JVM type system provides static guarantees.
- **Plan determinism**: Given the same world state, GOAP produces the same plan. Plans are auditable sequences.
- **Information flow bounds**: Data tainted as untrusted cannot reach sensitive sinks without explicit policy approval (from CaMeL's taint tracking).

### 9.2 Conditionally Guaranteed (Require Correct Implementation)

- **Control flow integrity**: Untrusted data cannot alter execution paths _if_ the interpreter correctly isolates P-LLM from untrusted content.
- **Noninterference**: Varying untrusted inputs produces equivalent authorized outputs _if_ the type system correctly propagates capabilities.
- **Complete mediation**: Every sensitive operation is checked _if_ all tools are registered and no unmediated execution paths exist.

### 9.3 Not Addressed

- **Scheming / intent control**: The architecture constrains what agents can _do_ but cannot detect scheming or strategic underperformance. Apollo/Redwood's control evaluation methodology remains necessary to bound P(harm | scheming).
- **Formal verification**: Current guarantees are empirical (AgentDojo benchmarks) or conditional on implementation correctness. Mechanized proofs for realistic systems don't exist yet.
- **The control window**: As Redwood Research emphasizes, control strategies work when the gap between trusted oversight and untrusted AI capabilities is manageable. The combined architecture extends this window but doesn't eliminate the fundamental scaling concern.

---

## 10. The Self-Fulfilling Prophecy Connection

Brief note, secondary to the main argument: structural guarantees also help with the adversarial dynamics problem.

If the architecture prevents bad actions, you don't need to _assume_ the model is trying them. This reduces the "adversarial relationship creating adversarial behavior" risk. Assume-adversarial for analysis ≠ implement-adversarial as relationship. Construction-based control lets you maintain the analytical assumption while deploying a less adversarial runtime.

---

## 11. Key Research Questions (Open)

1. **What is the minimum structure required on dynamically-generated actions such that planning over them preserves safety-relevant properties?**

2. **What is the coarsest action granularity that still provides meaningful safety properties?** Finer actions give more control surface but eventually you hit the "can't enumerate" wall.

3. **Under what conditions do T-generated schemas provide meaningful safety bounds?** Formalizing: if T correctly identifies the minimal capability set for task X, and U is constrained to that set, then U's maximum harm is bounded by [something analyzable].

4. **Can we characterize the constructible set?** Which tasks fall into the domain where construction-based control works, and what does the handoff to observation-based control look like at the boundary?

5. **Dynamic action spaces**: The constraint that matters isn't "compile-time" — it's "trusted enumeration vs. untrusted generation." Dynamic is fine if the action source is trusted.

6. **How does the CaMeL → bash/code execution mapping actually work?** This is the concrete gap. CaMeL solves prompt injection via data flow separation. What is the analogous structural constraint for the bash command / code execution setting from Greenblatt? This is where the argument either becomes concrete or stays hand-wavy.

---

## 12. Relationship to Existing Literature

### 12.1 Papers in Project Knowledge (Direct Relevance)

**Greenblatt et al. (2024) — "AI Control: Improving Safety Despite Intentional Subversion"**
The foundational AI control paper. Introduces the T/U (trusted/untrusted) framework, control evaluations, and the APPS backdooring setting. CaMeL+GOAP proposes a _structural_ complement to the _probabilistic_ monitoring approach developed here. The T/U split maps directly onto the schema generation architecture (Section 8.3), applied at the action-schema level rather than output monitoring.

**Griffin et al. (2024) — "Games for AI Control"**
Formalizes AI control as a multi-objective, partially observable, stochastic game. CaMeL+GOAP changes the game structure: if the action space is structurally bounded, the red team's strategy space is correspondingly constrained, potentially making optimal protocol synthesis more tractable. The safety-usefulness Pareto frontier should shift favorably for tasks in the constructible set.

**Factor(D,U) / Factor(U,T) (December 2025) — Factored Cognition for AI Control**
Applies factored cognition to the control setting: decompose tasks with one model, implement with another, monitor both layers. The headline finding is that monitoring _plans_ (abstract task decompositions) yields AUROC 0.52 — effectively chance — while monitoring _concrete implementations_ achieves AUROC 0.96. The authors frame this as evidence that monitors require concrete context.

_Our reframing_: The paper accidentally demonstrates the CaMeL+GOAP thesis more strongly than the authors realize. In Factor(D,U), the trusted decomposer constrains the attack surface available to the untrusted implementer — their own language is that "D imposes structural prevention: by generating only trusted child tasks, it severely constrains the attack surface available to the untrusted implementer." This is Tier 1 structural work. The monitor operating on implementations isn't doing the heavy lifting; it's mop-up duty after the architecture has already eliminated most attack vectors.

The AUROC 0.52 result is independently valuable as a concrete datapoint for monitoring degradation under abstraction — not a hypothetical distribution-shift argument but a demonstrated failure within the control literature itself. Factor(D,U) also occupies an interesting position relative to the constructible set boundary: the decomposer performs a structurally constraining role (Tier 1), while the implementer operates in the residual space where monitoring is still needed (Tier 3). This is exactly the hybrid architecture the three-tier framework argues for, instantiated by a paper that didn't frame it that way.

**Schaeffer et al. (February 2026) — "Attack Selection Reduces Safety in Concentrated Settings"**
Evaluates what happens when an adversarial untrusted model can _choose when to attack_ based on its assessment of monitor vulnerability. Safety drops from 99% to 59% when the attacker reasons about what the monitor will flag. The paper frames attack selection as a classification problem and shows safety is far more sensitive to FPR than TPR.

This is the strongest recent evidence for the uncomfortable implication (Section 7.4) — empirical monitoring degrades under adaptive adversaries, not just distribution shift. The 99% → 59% drop is a regime change, not marginal degradation. Construction-based control sidesteps this entirely for tasks in the constructible set: if the action space is structurally bounded, there is no "when to attack" decision — the attack surface doesn't vary with context because it's architecturally fixed. Attack selection is only a threat when the monitor is the primary safety mechanism, which is exactly the dependency the three-tier framework argues against.

**Ctrl-Z (Bhatt et al., 2025) — "Controlling AI Agents via Resampling"**
Extends control to multi-step sysadmin tasks using BashBench (257 tasks). Directly relevant as a benchmark and threat model for the sysadmin demonstration domain (Section 8). Their finding that multi-step attacks reduce per-step suspiciousness is exactly the kind of attack that construction-based control addresses differently — rather than detecting low-suspicion steps, you prevent unauthorized action sequences structurally. Resampling and CaMeL+GOAP address the same problem (multi-step attacks) from opposite paradigms (observation vs. construction).

**Mallen et al. (2025) — "Subversion Strategy Eval"**
Evaluates whether models can _statelessly strategize_ to subvert control protocols. CaMeL+GOAP reduces the strategy space available to a subversive model — if the action set is bounded and plan selection is deterministic, strategic reasoning about _which actions to take_ is moot. The remaining attack surface is within-action content manipulation, a narrower strategic challenge than full protocol subversion.

**AgentDojo (Debenedetti et al., 2024)**
The benchmark used to evaluate CaMeL. Provides empirical grounding for CaMeL's near-100% attack prevention claim. AgentDojo's task suites (banking, travel, workspace, Slack) overlap significantly with the constructible set where GOAP applies — structured workflows with known tools.

**SHADE Arena**
Evaluates sabotage and monitoring in LLM agents. Relevant as an evaluation framework for testing whether construction-based control actually prevents the sabotage patterns SHADE identifies.

**Wen et al. (2025) — "Adaptive Deployment of Untrusted LLMs Reduces Distributed Threats"**
Tested in the APPS setting with a distributed threat model. Introduced adaptive protocols that adjust monitoring intensity based on accumulated suspicion. Relevant as a Tier 3 refinement — better empirical monitoring — but doesn't address the structural question.

### 12.2 External Key References

- **Debenedetti et al. (2025)** — "Defeating Prompt Injections by Design" (CaMeL paper)
- **Tallam & Miller (2025)** — "Operationalizing CaMeL"
- **Willison (2023)** — "The Dual LLM pattern" (precursor to CaMeL)
- **Dalrymple et al.** — "Guaranteed Safe AI" framework (template for formal verification goals)
- **Invariant Labs (ICML 2024)** — Security rules in DSL, trace analysis before tool execution
- **Embabel documentation** — docs.embabel.com

### 12.3 The Case for Ensuring That Powerful AIs Are Controlled (Shlegeris & Greenblatt)

The strategic framing document for AI control. CaMeL+GOAP fits their argument that control strategies should be "robust to some employees intentionally subverting security measures" — the construction-based approach doesn't require trusting the model's outputs, only trusting the architectural constraints. The analogy to handling compromised employees maps directly: you don't need to know _if_ the model is scheming, you need to ensure the architecture limits what a schemer could accomplish.

---

## 13. Open Weaknesses to Address

- How much capability do structural constraints cost? CaMeL's 77% vs 84% is promising but domain-specific.
- Can GOAP-style planning handle the flexibility needed for real-world agentic tasks?
- Mutation testing overhead — is it practical at scale?
- Does this just push the problem to "who defines the action space / security policies"? (The specification problem: structural guarantees are only as good as the spec they enforce.)
- Does this extend beyond code execution to other control settings?

---

## 14. Strategic Context

### 14.1 Why This Matters for Daniel's Research Trajectory

- Demonstrates ability to design defenses, not just break them (the missing portfolio piece)
- "Blue-team-by-earned-paranoia" — defense-side work informed by deep red-team experience is the rarest and most valuable combination in the field
- Directly relevant to Apollo Research / Redwood Research positions
- Novel architectural contribution that shifts how the field thinks about the problem space

### 14.2 Possible Output Formats

**Option A: Short and punchy (2-3 pages)** — just the core argument, one example per tier, clear implications. Better for "send to Rob and Matthew."

**Option B: Full paper draft (8-12 pages)** — literature review, worked examples, formal-ish framework. Better for "submit somewhere" or "Redwood takes this seriously."

**Recommendation: Option A first, expand if it lands.**

### 14.3 Hackathon Decision (March 20-22, 2026)

Produce a hackathon-sized proof-of-concept or demo of this.

### 14.4 Paper-Shaped Abstract

> Not all agent tasks need open-ended reasoning, and for the ones that don't, we're paying a massive safety tax by using monitor-based control when construction-based control would give us stronger guarantees at lower cost. We characterize the set of tasks amenable to construction-based control, demonstrate a combined GOAP+CaMeL architecture on sysadmin tasks, show the safety properties obtained structurally, and red-team our own design to establish where the guarantees break down.

---

## 15. Conversation Sources

- **Embabel agent planning and GOAP algorithm** (Dec 17, 2025): Deep technical dive on GOAP algorithm, CaMeL synthesis report, sysadmin domain exploration, T/U schema architecture. [Link](https://claude.ai/chat/953246f8-3bbc-4db6-bc85-3c2bc9428600)
- **AI control hackathon 2026** (Feb 22, 2026): Hackathon scoping, "control by construction" articulation, boundary characterization, strategic framing for Buck. [Link](https://claude.ai/chat/1e39fa07-9051-4bc1-bacc-c592e225ed3f)
- **Code generation monitoring in AI control** (Feb 9, 2026): Post-Greenblatt literature survey, Factor(U,T) analysis, Schaeffer attack selection, monitoring landscape. [Link](https://claude.ai/chat/4cfa0c4d-8d32-4baa-bd46-e9511e85f481)
- **Senior Research Engineer opportunity at FAR.AI** (Mar 6, 2026): Writeup outline, Factor(D,U) reframing, three-tier framework development, SQL injection analogy. [Link](https://claude.ai/chat/0c2d2eb6-2a20-44f3-8930-af38d0771ffe)
- **Agent protocol comparative analysis** (Oct 2025): Early Embabel exploration. [Link](https://claude.ai/chat/a41351c3-8a62-4001-9750-a2637be85592)
