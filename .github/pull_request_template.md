# Pull Request Template

## 1. What issue does this PR solve?
- Please provide a clear and concise description of the issue being addressed. 
- Reference any related issue numbers (e.g., `Fixes #123`).

---

## 2. What/How is the solution implemented?
- Describe the solution and its implementation details.
- Include code snippets or diagrams if necessary for clarity.

---

## 3. References for the solution implemented
- List any documentation, articles, or resources that informed your solution.

---

## 4. Type of change

Please check the option that is related to your PR.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
  - In this case, we recommend to discuss your modification on GitHub issues before creating the PR
- [ ] Documentations (modification for documents)

---

## 5. Testing Details
- Make sure any user facing api's are not affected.
- Add relevant tests of input/output schema and validations so that no code changes should effect for any already exisiting apis.
- **Unit Tests:**  
    - [ ] Added test for training loop
    - [ ] Added test for inference
    - [ ] Added test for ensuring _ref and _fn perform similarly [Reference](https://github.com/state-spaces/mamba/blob/main/tests/ops/test_selective_scan.py)

---

## 6. Additional Details
- Include any other relevant information, such as:
  - Dependencies added or updated.
  - Known issues or caveats.
  - Future considerations or refactoring ideas.

---

## 7. Checklist
- [ ] My code follows [the style guidelines similar to this project](https://deepchem.readthedocs.io/en/latest/development_guide/coding.html)
  - [ ] Run `black . --line-length=79`
  - [ ] Run `isort . --profile=black`
  - [ ] Run `mypy -p lrnnx` and check no errors
  - [ ] Run `python -m doctest <modified file>` and check no errors
- [ ] All tests pass (if applicable).
- [ ] Documentation has been updated (if relevant).
- [ ] Changes have been communicated to the team.

---

Thank you for your contribution! Please ensure that your PR is aligned with the project guidelines and that you have filled out all relevant sections.
