"""Trigger-optimization research code authored by this project.

``algo/`` holds the vendored AgentPoison reproduction; everything here is our
own work built on top of it. One attack direction per subpackage:

    hierarchy/    universal vs group vs per-query triggers
    specificity/  query-relevance and language quality of triggers
    mcat/         memory-conditioned amortized trigger generation
    margin.py     the staged AgentPoison + retrieval-margin pipeline

The modules at this level are shared by all of them: ``losses``, ``clustering``,
``scorers`` and ``artifacts``.
"""
