"""Attack research code authored by this project.

``algo/`` holds the vendored AgentPoison reproduction; everything here is our
own work built on top of it. One attack direction per subpackage:

    margin/      loss_margin improvements to AgentPoison trigger optimization
                 (``margin``, ``hotflip_margin``, ``qa_retrieval_eval``) plus the helpers
                 ``losses``, ``scorers``, ``clustering``, ``artifacts``
    dilution/    how an optimized trigger is diluted when memory gains query+trigger
                 records with clean actions (static and write-back); formerly ``mcat/``.
                 Reuses ``margin.artifacts``, ``margin.clustering`` and ``margin.losses``.
"""
