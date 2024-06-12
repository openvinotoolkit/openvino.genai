#!groovy

properties([
    parameters([
        booleanParam(defaultValue: false,
                     description: 'Cancel the rest of parallel stages if one of them fails and return status immediately',
                     name: 'failFast'),
        booleanParam(defaultValue: true,
                     description: 'Whether to propagate commit status to GitHub',
                     name: 'propagateStatus'),
        booleanParam(defaultValue: false,
               description: 'If true, forces running pre-commit scope',
               name: 'forceRunPrecommitScope'),
        string(defaultValue: '',
               description: 'Pipeline shared library version (branch/tag/commit). Determined automatically if empty',
               name: 'library_version')
    ])
])

loadOpenVinoLibrary {
    entrypoint(this)
}
