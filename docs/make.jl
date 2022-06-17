using Documenter, DynamicNLPModels

const _PAGES = [
    "Introduction" => "index.md",
    "Quick Start"=>"guide.md",
    "API Manual" => "api.md"
]


makedocs(
    sitename = "DynamicNLPModels",
    authors = "David Cole, Sungho Shin, Francois Pacaud",
    format = Documenter.LaTeX(platform="docker"),
    pages = _PAGES
)

makedocs(
    sitename = "DynamicNLPModels",
    modules = [DynamicNLPModels],
    authors = "David Cole, Sungho Shin, Francois Pacaud",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        sidebar_sitename = true,
        collapselevel = 1,
    ),
    pages = _PAGES,
    clean = false,
)


deploydocs(
    repo = "github.com/MadNLP/DynamicNLPModels.jl.git",
    target = "build",
    devbranch = "main",
    devurl = "dev",
    push_preview = true,
)

