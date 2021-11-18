lock_dependencies: ## Copy dependencies defined in pyproject.toml to requirements.txt
	poetry export -f requirements.txt > requirements.txt