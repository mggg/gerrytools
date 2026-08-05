# TODO

- [ ] Ben module
  - [x] Improve integration with binary-ensemble
  - [ ] Add some python bindings for the MSMS parser and the SMC parser
    - [ ] In the SMC parser and runner, maybe just save in the RDS format and then do the
          conversion on top of that.
    - [ ] In MSMS, allow for arbitrary numbers of levels to be parsed

- [ ] Data Module
  - [ ] Add some fallback mirrors for the data (e.g. Chicago's mirrors)
  - [ ] Change the way that we estimate CVAP (WolfRam?)

- [ ] Scoring Module
  - [ ] Support a void district label in `RegionParts` so incomplete plans exclude unassigned
        nodes (districtr)
  - [ ] Return part counts by region label instead of only the aggregate `RegionParts` total
        (districtr)
  - [ ] Support preparing a child-only evaluator from a hybrid parent/child graph without copying
        the child graph, for persistent service evaluators (districtr)
