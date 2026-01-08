# Only compile this module if hf_hub is available
# Add {:hf_hub, "~> 0.1.3"} to your deps to enable HuggingFace features
if Code.ensure_loaded?(HfHub.Api) do
  defmodule Ollixir.HuggingFace do
    @moduledoc """
      HuggingFace Hub integration for Ollixir.

      > #### Optional Dependency {: .info}
      >
      > This module requires the `hf_hub` package. Add it to your dependencies:
      >
      >     {:hf_hub, "~> 0.1.3"}
      >
      > The module will not be available if `hf_hub` is not installed.

      This module provides seamless integration with HuggingFace Hub, enabling you to:

    - Discover GGUF model files in HuggingFace repositories
    - Auto-select optimal quantization based on preferences
    - Build Ollama-compatible model references
    - Pull and run HuggingFace models directly through Ollama

    ## Overview

    Ollama natively supports running GGUF models from HuggingFace Hub using the
    `hf.co/{username}/{repository}:{quantization}` model reference format. This module
    adds discovery and convenience features on top of that capability.

    ## Quick Start

        # Initialize Ollixir client
        client = Ollixir.init()

        # Discover available GGUF files
        {:ok, ggufs} = Ollixir.HuggingFace.list_gguf_files("bartowski/Llama-3.2-1B-Instruct-GGUF")

        # Auto-select best quantization
        {:ok, model_ref, info} = Ollixir.HuggingFace.auto_select("bartowski/Llama-3.2-1B-Instruct-GGUF")

        # Pull and chat
        {:ok, _} = Ollixir.HuggingFace.pull(client, "bartowski/Llama-3.2-1B-Instruct-GGUF", quantization: "Q4_K_M")
        {:ok, response} = Ollixir.HuggingFace.chat(client, "bartowski/Llama-3.2-1B-Instruct-GGUF",
          [%{role: "user", content: "Hello!"}],
          quantization: "Q4_K_M"
        )

    ## Direct Usage (No Discovery)

    If you already know the repository and quantization you want, you can skip
    this module entirely and use Ollixir directly:

        Ollixir.chat(client,
          model: "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
          messages: [%{role: "user", content: "Hello!"}]
        )

    ## Quantization Selection

    The module uses a preference order optimized for quality/size balance:

    1. Q4_K_M, Q4_K_S (best balance for most users)
    2. Q5_K_M, Q5_K_S (higher quality, larger size)
    3. Q6_K, Q8_0 (even higher quality)
    4. IQ4_XS, IQ3_M (smaller, for constrained environments)
    5. F16, BF16 (full precision, largest)

    You can also specify your own preference or filter by maximum size.
    """

    @typedoc """
    Information about a GGUF file in a HuggingFace repository.
    """
    @type gguf_info :: %{
            filename: String.t(),
            size_bytes: non_neg_integer(),
            size_gb: float(),
            quantization: String.t(),
            ollama_tag: String.t()
          }

    @typedoc """
    Options for HuggingFace operations.
    """
    @type hf_opts :: [
            quantization: String.t(),
            revision: String.t(),
            token: String.t()
          ]

    # Default quantization preference order (quality vs size tradeoff)
    # Q4_K_M is the sweet spot for most users
    @quant_preference ~w(
    Q4_K_M Q4_K_S Q4_K Q4_K_L
    Q5_K_M Q5_K_S Q5_K Q5_K_L
    Q6_K Q6_K_L
    Q8_0
    IQ4_XS IQ4_NL
    IQ3_M IQ3_S IQ3_XS IQ3_XXS
    Q4_0 Q5_0 Q5_1
    Q3_K_M Q3_K_S Q3_K_L Q3_K_XL
    F16 BF16 F32
  )

    # ============================================================================
    # Model Reference Building
    # ============================================================================

    @doc """
    Builds an Ollama model reference from a HuggingFace repository ID.

    Ollama natively supports HuggingFace models using the format:
    `hf.co/{username}/{repository}:{quantization}`

    ## Parameters

      - `repo_id` - HuggingFace repository ID (e.g., "bartowski/Llama-3.2-1B-Instruct-GGUF")
      - `opts` - Options:
        - `:quantization` - Quantization tag (e.g., "Q4_K_M", "IQ3_M")

    ## Examples

        iex> Ollixir.HuggingFace.model_ref("bartowski/Llama-3.2-1B-Instruct-GGUF")
        "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

        iex> Ollixir.HuggingFace.model_ref("bartowski/Llama-3.2-1B-Instruct-GGUF", quantization: "Q8_0")
        "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0"

    """
    @spec model_ref(String.t(), keyword()) :: String.t()
    def model_ref(repo_id, opts \\ []) do
      base = "hf.co/#{repo_id}"
      quant = opts[:quantization]

      if quant do
        "#{base}:#{quant}"
      else
        base
      end
    end

    @doc """
    Parses an Ollama HuggingFace model reference into its components.

    ## Examples

        iex> Ollixir.HuggingFace.parse_model_ref("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
        {:ok, %{repo_id: "bartowski/Llama-3.2-1B-Instruct-GGUF", quantization: "Q4_K_M"}}

        iex> Ollixir.HuggingFace.parse_model_ref("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
        {:ok, %{repo_id: "bartowski/Llama-3.2-1B-Instruct-GGUF", quantization: nil}}

        iex> Ollixir.HuggingFace.parse_model_ref("llama3.2")
        {:error, :not_hf_model}

    """
    @spec parse_model_ref(String.t()) ::
            {:ok, %{repo_id: String.t(), quantization: String.t() | nil}}
            | {:error, :not_hf_model}
    def parse_model_ref(model_ref) do
      case Regex.run(~r/^hf\.co\/(.+?)(?::(.+))?$/, model_ref) do
        [_, repo_id, quant] -> {:ok, %{repo_id: repo_id, quantization: quant}}
        [_, repo_id] -> {:ok, %{repo_id: repo_id, quantization: nil}}
        _ -> {:error, :not_hf_model}
      end
    end

    @doc """
    Checks if a model reference is a HuggingFace model.

    ## Examples

        iex> Ollixir.HuggingFace.hf_model?("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
        true

        iex> Ollixir.HuggingFace.hf_model?("llama3.2")
        false

    """
    @spec hf_model?(String.t()) :: boolean()
    def hf_model?(model_ref) do
      String.starts_with?(model_ref, "hf.co/") or
        String.starts_with?(model_ref, "huggingface.co/")
    end

    # ============================================================================
    # GGUF Discovery
    # ============================================================================

    @doc """
    Lists all GGUF files in a HuggingFace repository.

    Uses the HuggingFace Hub API to discover available GGUF model files,
    extracting quantization type and file size for each.

    ## Parameters

      - `repo_id` - HuggingFace repository ID
      - `opts` - Options passed to `HfHub.Api.list_repo_tree/2`:
        - `:revision` - Git revision (default: "main")
        - `:token` - HuggingFace API token for private repos

    ## Returns

    A list of maps containing:
      - `:filename` - Full filename (e.g., "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
      - `:size_bytes` - File size in bytes
      - `:size_gb` - File size in gigabytes (rounded to 2 decimal places)
      - `:quantization` - Extracted quantization type (e.g., "Q4_K_M")
      - `:ollama_tag` - The tag to use with Ollama (uppercase quantization)

    ## Examples

        {:ok, ggufs} = Ollixir.HuggingFace.list_gguf_files("bartowski/Llama-3.2-1B-Instruct-GGUF")
        # => [
        #   %{filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf", size_gb: 0.75, quantization: "Q4_K_M", ...},
        #   %{filename: "Llama-3.2-1B-Instruct-Q8_0.gguf", size_gb: 1.23, quantization: "Q8_0", ...},
        #   ...
        # ]

    """
    @spec list_gguf_files(String.t(), keyword()) :: {:ok, [gguf_info()]} | {:error, term()}
    def list_gguf_files(repo_id, opts \\ []) do
      tree_opts = Keyword.merge([recursive: true, expand: true], opts)

      case HfHub.Api.list_repo_tree(repo_id, tree_opts) do
        {:ok, entries} ->
          gguf_files =
            entries
            |> Enum.filter(fn entry ->
              entry.type == :file && String.ends_with?(entry.path, ".gguf")
            end)
            |> Enum.map(fn entry ->
              size = entry.size || get_in(entry, [:lfs, "size"]) || 0
              quant = extract_quantization(entry.path)

              %{
                filename: entry.path,
                size_bytes: size,
                size_gb: Float.round(size / 1_073_741_824, 2),
                quantization: quant,
                ollama_tag: String.upcase(quant)
              }
            end)
            |> Enum.sort_by(& &1.size_bytes)

          {:ok, gguf_files}

        {:error, reason} ->
          {:error, reason}
      end
    end

    @doc """
    Extracts the quantization type from a GGUF filename.

    Parses common quantization patterns from filenames like:
    - `Llama-3.2-1B-Instruct-Q4_K_M.gguf` -> "Q4_K_M"
    - `model-IQ3_M.gguf` -> "IQ3_M"
    - `model-Q6_K.gguf` -> "Q6_K"
    - `model-f16.gguf` -> "F16"

    ## Examples

        iex> Ollixir.HuggingFace.extract_quantization("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        "Q4_K_M"

        iex> Ollixir.HuggingFace.extract_quantization("model-IQ3_M.gguf")
        "IQ3_M"

        iex> Ollixir.HuggingFace.extract_quantization("model-Q6_K.gguf")
        "Q6_K"

        iex> Ollixir.HuggingFace.extract_quantization("unknown-format.gguf")
        "unknown"

    """
    @spec extract_quantization(String.t()) :: String.t()
    def extract_quantization(filename) do
      # Common quantization patterns in GGUF filenames
      # Order matters - more specific patterns first
      patterns = [
        # K-quants with size suffix: Q4_K_M, Q5_K_S, Q3_K_L, Q3_K_XL, Q6_K_L
        ~r/[_-](Q\d+_K_[SMLX]+)/i,
        # K-quants without suffix: Q6_K, Q4_K
        ~r/[_-](Q\d+_K)(?:[._-]|\.gguf)/i,
        # Standard quants with variant: Q4_0_4_4, Q4_0_4_8, Q4_0_8_8
        ~r/[_-](Q\d+_\d+_\d+_\d+)/i,
        # Standard quants: Q4_0, Q8_0, Q5_1
        ~r/[_-](Q\d+_\d+)/i,
        # I-quants: IQ3_M, IQ4_XS, IQ2_XXS
        ~r/[_-](IQ\d+_[A-Z]+)/i,
        # Float types: F16, F32, BF16
        ~r/[_-](B?F\d+)/i
      ]

      Enum.find_value(patterns, "unknown", fn pattern ->
        case Regex.run(pattern, filename, capture: :all_but_first) do
          [quant] -> String.upcase(quant)
          _ -> nil
        end
      end)
    end

    # ============================================================================
    # Quantization Selection
    # ============================================================================

    @doc """
    Returns the default quantization preference order.

    This is the order used by `best_quantization/1` and `auto_select/2` when
    choosing the optimal quantization for a model.

    ## Examples

        Ollixir.HuggingFace.quant_preference()
        # => ["Q4_K_M", "Q4_K_S", "Q4_K", "Q4_K_L", "Q5_K_M", ...]

    """
    @spec quant_preference() :: [String.t()]
    def quant_preference, do: @quant_preference

    @doc """
    Finds the best available quantization from a list of GGUF files.

    Uses the default preference order to select the highest-priority
    quantization that is available in the given list.

    ## Parameters

      - `gguf_files` - List of GGUF info maps from `list_gguf_files/2`
      - `opts` - Options:
        - `:preference` - Custom preference list (default: `quant_preference/0`)
        - `:max_size_gb` - Maximum file size in GB (filters out larger files)

    ## Examples

        {:ok, ggufs} = Ollixir.HuggingFace.list_gguf_files("bartowski/Llama-3.2-1B-Instruct-GGUF")

        Ollixir.HuggingFace.best_quantization(ggufs)
        # => "Q4_K_M"

        Ollixir.HuggingFace.best_quantization(ggufs, max_size_gb: 1.0)
        # => "Q4_K_M" (if under 1GB) or next smallest

    """
    @spec best_quantization([gguf_info()], keyword()) :: String.t() | nil
    def best_quantization(gguf_files, opts \\ []) do
      preference = Keyword.get(opts, :preference, @quant_preference)
      max_size = Keyword.get(opts, :max_size_gb)

      # Filter by size if specified
      filtered =
        if max_size do
          Enum.filter(gguf_files, &(&1.size_gb <= max_size))
        else
          gguf_files
        end

      available = MapSet.new(filtered, & &1.quantization)

      Enum.find(preference, fn q ->
        MapSet.member?(available, q)
      end) || List.first(filtered)[:quantization]
    end

    @doc """
    Auto-selects the best model from a HuggingFace repository.

    Discovers available GGUF files and selects the optimal quantization
    based on the preference order.

    ## Parameters

      - `repo_id` - HuggingFace repository ID
      - `opts` - Options:
        - `:quantization` - Force a specific quantization instead of auto-selecting
        - `:max_size_gb` - Maximum file size in GB
        - `:revision` - Git revision (default: "main")
        - `:token` - HuggingFace API token

    ## Returns

    A tuple of `{:ok, model_ref, gguf_info}` where:
      - `model_ref` - The full Ollama model reference (e.g., "hf.co/repo:Q4_K_M")
      - `gguf_info` - The selected GGUF file info map

    ## Examples

        {:ok, model_ref, info} = Ollixir.HuggingFace.auto_select("bartowski/Llama-3.2-1B-Instruct-GGUF")
        # => {:ok, "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", %{quantization: "Q4_K_M", ...}}

        # With size constraint
        {:ok, model_ref, info} = Ollixir.HuggingFace.auto_select("bartowski/Llama-3.2-1B-Instruct-GGUF",
          max_size_gb: 0.7
        )

        # Force specific quantization
        {:ok, model_ref, info} = Ollixir.HuggingFace.auto_select("bartowski/Llama-3.2-1B-Instruct-GGUF",
          quantization: "Q8_0"
        )

    """
    @spec auto_select(String.t(), keyword()) ::
            {:ok, String.t(), gguf_info()} | {:error, term()}
    def auto_select(repo_id, opts \\ []) do
      hf_opts = Keyword.take(opts, [:revision, :token])

      with {:ok, ggufs} <- list_gguf_files(repo_id, hf_opts) do
        if Enum.empty?(ggufs) do
          {:error, :no_gguf_files}
        else
          preferred = opts[:quantization]
          max_size = opts[:max_size_gb]

          selected =
            cond do
              # User specified a quantization
              preferred ->
                target = String.upcase(preferred)
                Enum.find(ggufs, fn g -> g.quantization == target end) || List.first(ggufs)

              # Auto-select with optional size constraint
              true ->
                best = best_quantization(ggufs, max_size_gb: max_size)
                Enum.find(ggufs, fn g -> g.quantization == best end)
            end

          if selected do
            model = model_ref(repo_id, quantization: selected.quantization)
            {:ok, model, selected}
          else
            {:error, :no_matching_quantization}
          end
        end
      end
    end

    # ============================================================================
    # Ollama Operations with HuggingFace Models
    # ============================================================================

    @doc """
    Pulls a HuggingFace model through Ollama.

    This is a convenience wrapper around `Ollixir.pull_model/2` that builds
    the correct model reference format.

    ## Parameters

      - `client` - Ollixir client from `Ollixir.init/1`
      - `repo_id` - HuggingFace repository ID
      - `opts` - Options:
        - `:quantization` - Quantization tag (recommended)
        - `:stream` - Stream progress updates (default: false)
        - Other options passed to `Ollixir.pull_model/2`

    ## Examples

        client = Ollixir.init()

        # Pull specific quantization
        {:ok, response} = Ollixir.HuggingFace.pull(client, "bartowski/Llama-3.2-1B-Instruct-GGUF",
          quantization: "Q4_K_M"
        )

        # Pull with streaming progress
        {:ok, stream} = Ollixir.HuggingFace.pull(client, "bartowski/Llama-3.2-1B-Instruct-GGUF",
          quantization: "Q4_K_M",
          stream: true
        )
        Enum.each(stream, &IO.inspect/1)

    """
    @spec pull(Ollixir.client(), String.t(), keyword()) :: {:ok, term()} | {:error, term()}
    def pull(client, repo_id, opts \\ []) do
      {hf_opts, pull_opts} = Keyword.split(opts, [:quantization])
      model = model_ref(repo_id, hf_opts)

      Ollixir.pull_model(client, [{:name, model} | pull_opts])
    end

    @doc """
    Chats with a HuggingFace model through Ollama.

    This is a convenience wrapper around `Ollixir.chat/2` that builds
    the correct model reference format.

    ## Parameters

      - `client` - Ollixir client from `Ollixir.init/1`
      - `repo_id` - HuggingFace repository ID
      - `messages` - List of message maps with `:role` and `:content`
      - `opts` - Options:
        - `:quantization` - Quantization tag (recommended)
        - `:stream` - Stream responses (default: false)
        - Other options passed to `Ollixir.chat/2`

    ## Examples

        client = Ollixir.init()

        {:ok, response} = Ollixir.HuggingFace.chat(client, "bartowski/Llama-3.2-1B-Instruct-GGUF",
          [%{role: "user", content: "Hello!"}],
          quantization: "Q4_K_M"
        )

        IO.puts(response["message"]["content"])

        # With streaming
        {:ok, stream} = Ollixir.HuggingFace.chat(client, "bartowski/Llama-3.2-1B-Instruct-GGUF",
          [%{role: "user", content: "Tell me a story"}],
          quantization: "Q4_K_M",
          stream: true
        )
        Enum.each(stream, fn chunk ->
          if content = get_in(chunk, ["message", "content"]), do: IO.write(content)
        end)

    """
    @spec chat(Ollixir.client(), String.t(), list(map()), keyword()) ::
            {:ok, term()} | {:error, term()}
    def chat(client, repo_id, messages, opts \\ []) do
      {hf_opts, chat_opts} = Keyword.split(opts, [:quantization])
      model = model_ref(repo_id, hf_opts)

      Ollixir.chat(client, [{:model, model}, {:messages, messages} | chat_opts])
    end

    @doc """
    Generates a completion from a HuggingFace model through Ollama.

    This is a convenience wrapper around `Ollixir.generate/2` (or `Ollixir.completion/2`)
    that builds the correct model reference format.

    ## Parameters

      - `client` - Ollixir client from `Ollixir.init/1`
      - `repo_id` - HuggingFace repository ID
      - `prompt` - The prompt string
      - `opts` - Options:
        - `:quantization` - Quantization tag (recommended)
        - `:stream` - Stream responses (default: false)
        - Other options passed to `Ollixir.generate/2`

    ## Examples

        client = Ollixir.init()

        {:ok, response} = Ollixir.HuggingFace.generate(client, "bartowski/Llama-3.2-1B-Instruct-GGUF",
          "Once upon a time",
          quantization: "Q4_K_M"
        )

        IO.puts(response["response"])

    """
    @spec generate(Ollixir.client(), String.t(), String.t(), keyword()) ::
            {:ok, term()} | {:error, term()}
    def generate(client, repo_id, prompt, opts \\ []) do
      {hf_opts, gen_opts} = Keyword.split(opts, [:quantization])
      model = model_ref(repo_id, hf_opts)

      Ollixir.generate(client, [{:model, model}, {:prompt, prompt} | gen_opts])
    end

    @doc """
    Generates embeddings from a HuggingFace model through Ollama.

    This is a convenience wrapper around `Ollixir.embed/2` that builds
    the correct model reference format.

    ## Parameters

      - `client` - Ollixir client from `Ollixir.init/1`
      - `repo_id` - HuggingFace repository ID (must be an embedding model)
      - `input` - Text or list of texts to embed
      - `opts` - Options:
        - `:quantization` - Quantization tag (recommended)
        - Other options passed to `Ollixir.embed/2`

    ## Examples

        client = Ollixir.init()

        {:ok, response} = Ollixir.HuggingFace.embed(client, "nomic-ai/nomic-embed-text-v1.5-GGUF",
          "Hello world",
          quantization: "Q4_K_M"
        )

        embeddings = response["embeddings"]

    """
    @spec embed(Ollixir.client(), String.t(), String.t() | [String.t()], keyword()) ::
            {:ok, term()} | {:error, term()}
    def embed(client, repo_id, input, opts \\ []) do
      {hf_opts, embed_opts} = Keyword.split(opts, [:quantization])
      model = model_ref(repo_id, hf_opts)

      Ollixir.embed(client, [{:model, model}, {:input, input} | embed_opts])
    end

    # ============================================================================
    # Model Info
    # ============================================================================

    @doc """
    Gets model information from HuggingFace Hub.

    Returns metadata about the model including downloads, tags, and file list.

    ## Parameters

      - `repo_id` - HuggingFace repository ID
      - `opts` - Options passed to `HfHub.Api.model_info/2`

    ## Examples

        {:ok, info} = Ollixir.HuggingFace.model_info("bartowski/Llama-3.2-1B-Instruct-GGUF")
        IO.puts("Downloads: \#{info.downloads}")
        IO.puts("Tags: \#{Enum.join(info.tags, ", ")}")

    """
    @spec model_info(String.t(), keyword()) :: {:ok, map()} | {:error, term()}
    def model_info(repo_id, opts \\ []) do
      HfHub.Api.model_info(repo_id, opts)
    end
  end
end
