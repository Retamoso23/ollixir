defmodule Ollixir.HuggingFaceIntegrationTest do
  use ExUnit.Case, async: false

  # These tests require network access to HuggingFace Hub
  # Run with: mix test --include hf_integration
  @moduletag :hf_integration

  alias Ollixir.HuggingFace

  # ============================================================================
  # HuggingFace Hub API Integration Tests
  # ============================================================================

  describe "list_gguf_files/2" do
    @tag timeout: 30_000
    test "lists GGUF files from bartowski repository" do
      {:ok, ggufs} = HuggingFace.list_gguf_files("bartowski/Llama-3.2-1B-Instruct-GGUF")

      assert is_list(ggufs)
      assert length(ggufs) > 0

      # Check structure of returned items
      [first | _] = ggufs
      assert is_binary(first.filename)
      assert String.ends_with?(first.filename, ".gguf")
      assert is_integer(first.size_bytes)
      assert is_float(first.size_gb)
      assert is_binary(first.quantization)
      assert is_binary(first.ollama_tag)
    end

    @tag timeout: 30_000
    test "returns error for non-existent repository" do
      result = HuggingFace.list_gguf_files("this-repo-does-not-exist-12345/fake-model")

      assert {:error, _reason} = result
    end

    @tag timeout: 30_000
    test "extracts quantizations correctly from real files" do
      {:ok, ggufs} = HuggingFace.list_gguf_files("bartowski/Llama-3.2-1B-Instruct-GGUF")

      # Should have various quantizations
      quants = Enum.map(ggufs, & &1.quantization) |> Enum.uniq()

      # Should have at least some recognized quantizations
      recognized = Enum.reject(quants, &(&1 == "unknown"))
      assert length(recognized) > 0

      # Common ones should be present
      assert "Q4_K_M" in quants or "Q4_K_S" in quants
    end
  end

  describe "auto_select/2" do
    @tag timeout: 30_000
    test "auto-selects best quantization" do
      {:ok, model_ref, info} = HuggingFace.auto_select("bartowski/Llama-3.2-1B-Instruct-GGUF")

      assert String.starts_with?(model_ref, "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:")
      assert is_map(info)
      assert is_binary(info.quantization)
    end

    @tag timeout: 30_000
    test "respects quantization option" do
      {:ok, model_ref, info} =
        HuggingFace.auto_select("bartowski/Llama-3.2-1B-Instruct-GGUF", quantization: "Q8_0")

      assert String.ends_with?(model_ref, ":Q8_0")
      assert info.quantization == "Q8_0"
    end

    @tag timeout: 30_000
    test "respects max_size_gb option" do
      {:ok, _model_ref, info} =
        HuggingFace.auto_select("bartowski/Llama-3.2-1B-Instruct-GGUF", max_size_gb: 0.7)

      # Selected model should be under 0.7 GB
      assert info.size_gb <= 0.7
    end

    @tag timeout: 30_000
    test "returns error for non-existent repository" do
      result = HuggingFace.auto_select("this-repo-does-not-exist-12345/fake-model")

      assert {:error, _reason} = result
    end
  end

  describe "model_info/2" do
    @tag timeout: 30_000
    test "retrieves model info from HuggingFace" do
      {:ok, info} = HuggingFace.model_info("bartowski/Llama-3.2-1B-Instruct-GGUF")

      assert info.id == "bartowski/Llama-3.2-1B-Instruct-GGUF"
      assert is_integer(info.downloads)
      assert is_list(info.tags)
    end

    @tag timeout: 30_000
    test "returns error for non-existent model" do
      result = HuggingFace.model_info("this-repo-does-not-exist-12345/fake-model")

      assert {:error, _reason} = result
    end
  end

  # ============================================================================
  # Ollama Integration Tests (require running Ollama server)
  # ============================================================================

  describe "Ollama operations" do
    # These tests require Ollama to be running
    # Run with: mix test --include ollama_live

    @describetag :ollama_live

    setup do
      client = Ollixir.init()
      {:ok, client: client}
    end

    @tag timeout: 300_000
    test "pull/3 downloads HF model", %{client: client} do
      # Use a small model for testing
      result =
        HuggingFace.pull(client, "bartowski/Llama-3.2-1B-Instruct-GGUF", quantization: "IQ4_XS")

      assert {:ok, _response} = result
    end

    @tag timeout: 60_000
    test "chat/4 works with HF model", %{client: client} do
      # Assumes model was already pulled
      result =
        HuggingFace.chat(
          client,
          "bartowski/Llama-3.2-1B-Instruct-GGUF",
          [%{role: "user", content: "Say hello"}],
          quantization: "IQ4_XS"
        )

      assert {:ok, response} = result
      assert get_in(response, ["message", "content"]) |> is_binary()
    end

    @tag timeout: 60_000
    test "generate/4 works with HF model", %{client: client} do
      # Assumes model was already pulled
      result =
        HuggingFace.generate(
          client,
          "bartowski/Llama-3.2-1B-Instruct-GGUF",
          "Hello",
          quantization: "IQ4_XS"
        )

      assert {:ok, response} = result
      assert response["response"] |> is_binary()
    end
  end
end
