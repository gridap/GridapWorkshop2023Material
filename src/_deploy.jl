using Literate
## include Literate scripts starting with following letters in the deploy
incl = ""
## Set `sol=true` to produce output with solutions contained and hints scripts. Otherwise the other way around.
sol = true
##
notebooks_folder = "../notebooks"
scripts_folder = "../scripts"

function replace_string(str)
        strn = str
        for st in ["./figures/" => "../assets/literate_figures/"]
            strn = replace(strn, st)
        end
    return strn
end

"""
    process_hashtag(str, hashtag, fn; striptag=true)

Process all lines in str which start with a hashtag by the
function `fn` (`line->newline`).

```
# drop lines starting with "#sol"
drop_sol = str -> process_hashtag(str, "#sol", line -> "")
Literate.notebook(fl, "notebooks", preproces=drop_sol)
```
"""
function process_hashtag(str, hashtag, fn; striptag=true)
    hashtag = strip(hashtag)
    occursin("\r\n", str) && error("""DOS line endings "\r"n" not supported""")
    out = ""
    regex = Regex(hashtag)
    for line in split(str, '\n')
        # line = if startswith(lstrip(line), hashtag)
        line = if occursin(regex, line)
            fn(striptag ? replace(line, hashtag=>"") : line)
        else
            line = line * "\n"
        end
        out = out * line
    end
    return out
end

"Use as `preproces` function to remove `#sol`-lines & just remote `#tag`-tag"
function rm_sol(str)
    str = process_hashtag(str, "#sol=", line->"")
    str = process_hashtag(str, "#hint=", line->"#" * line * "\n")
    return str
end
"Use as `preproces` function to remove `#hint`-lines & just remote `#sol`-tag"
function rm_hint(str)
    str = process_hashtag(str, "#sol=", line->line * "\n")
    str = process_hashtag(str, "#hint=", line->"")
    return str
end

for fl in readdir()
    if splitext(fl)[end]!=".jl" || splitpath(@__FILE__)[end]==fl || !occursin(incl, fl)
        continue
    end

    # Render
    Literate.notebook(fl, notebooks_folder, credit=false, execute=false, mdstrings=true, preprocess=rm_sol)
    Literate.script(fl, scripts_folder, credit=false, execute=false, mdstrings=true, preprocess=rm_sol)
    if sol
        notebooks_sol_folder = string(notebooks_folder,"-solutions")
        scripts_sol_folder = string(scripts_folder,"-solutions")
        Literate.notebook(fl, notebooks_sol_folder, credit=false, execute=false, mdstrings=true, preprocess=rm_hint)
        Literate.script(fl, scripts_sol_folder, credit=false, execute=false, mdstrings=true, preprocess=rm_hint)
    end
end