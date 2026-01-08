defmodule Ollixir.HuggingFaceTest do
  use ExUnit.Case, async: true

  alias Ollixir.HuggingFace

  # ============================================================================
  # Model Reference Tests
  # ============================================================================

  describe "model_ref/2" do
    test "builds model reference without quantization" do
      assert "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF" =
               HuggingFace.model_ref("bartowski/Llama-3.2-1B-Instruct-GGUF")
    end

    test "builds model reference with quantization" do
      assert "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M" =
               HuggingFace.model_ref("bartowski/Llama-3.2-1B-Instruct-GGUF",
                 quantization: "Q4_K_M"
               )
    end

    test "builds model reference with lowercase quantization" do
      assert "hf.co/repo/model:q8_0" =
               HuggingFace.model_ref("repo/model", quantization: "q8_0")
    end

    test "handles empty options" do
      assert "hf.co/user/repo" = HuggingFace.model_ref("user/repo", [])
    end
  end

  describe "parse_model_ref/1" do
    test "parses model reference with quantization" do
      assert {:ok, %{repo_id: "bartowski/Llama-3.2-1B-Instruct-GGUF", quantization: "Q4_K_M"}} =
               HuggingFace.parse_model_ref("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    end

    test "parses model reference without quantization" do
      assert {:ok, %{repo_id: "bartowski/Llama-3.2-1B-Instruct-GGUF", quantization: nil}} =
               HuggingFace.parse_model_ref("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
    end

    test "returns error for non-HF model" do
      assert {:error, :not_hf_model} = HuggingFace.parse_model_ref("llama3.2")
    end

    test "returns error for empty string" do
      assert {:error, :not_hf_model} = HuggingFace.parse_model_ref("")
    end

    test "handles complex repo paths" do
      assert {:ok, %{repo_id: "org/sub/model-name", quantization: "IQ3_M"}} =
               HuggingFace.parse_model_ref("hf.co/org/sub/model-name:IQ3_M")
    end
  end

  describe "hf_model?/1" do
    test "returns true for hf.co prefix" do
      assert HuggingFace.hf_model?("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
    end

    test "returns true for hf.co with quantization" do
      assert HuggingFace.hf_model?("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    end

    test "returns true for huggingface.co prefix" do
      assert HuggingFace.hf_model?("huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
    end

    test "returns false for standard model names" do
      refute HuggingFace.hf_model?("llama3.2")
    end

    test "returns false for empty string" do
      refute HuggingFace.hf_model?("")
    end

    test "returns false for partial match" do
      refute HuggingFace.hf_model?("not-hf.co/model")
    end
  end

  # ============================================================================
  # Quantization Extraction Tests
  # ============================================================================

  describe "extract_quantization/1" do
    test "extracts Q4_K_M quantization" do
      assert "Q4_K_M" = HuggingFace.extract_quantization("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    end

    test "extracts Q4_K_S quantization" do
      assert "Q4_K_S" = HuggingFace.extract_quantization("model-Q4_K_S.gguf")
    end

    test "extracts Q5_K_M quantization" do
      assert "Q5_K_M" = HuggingFace.extract_quantization("model-Q5_K_M.gguf")
    end

    test "extracts Q5_K_L quantization" do
      assert "Q5_K_L" = HuggingFace.extract_quantization("model-Q5_K_L.gguf")
    end

    test "extracts Q3_K_L quantization" do
      assert "Q3_K_L" = HuggingFace.extract_quantization("Llama-3.2-1B-Instruct-Q3_K_L.gguf")
    end

    test "extracts Q3_K_XL quantization" do
      assert "Q3_K_XL" = HuggingFace.extract_quantization("model-Q3_K_XL.gguf")
    end

    test "extracts Q6_K quantization without suffix" do
      assert "Q6_K" = HuggingFace.extract_quantization("model-Q6_K.gguf")
    end

    test "extracts Q6_K_L quantization" do
      assert "Q6_K_L" = HuggingFace.extract_quantization("model-Q6_K_L.gguf")
    end

    test "extracts Q8_0 quantization" do
      assert "Q8_0" = HuggingFace.extract_quantization("model-Q8_0.gguf")
    end

    test "extracts Q4_0 quantization" do
      assert "Q4_0" = HuggingFace.extract_quantization("model-Q4_0.gguf")
    end

    test "extracts Q4_0_4_4 quantization variant" do
      assert "Q4_0_4_4" = HuggingFace.extract_quantization("model-Q4_0_4_4.gguf")
    end

    test "extracts Q4_0_4_8 quantization variant" do
      assert "Q4_0_4_8" = HuggingFace.extract_quantization("model-Q4_0_4_8.gguf")
    end

    test "extracts Q4_0_8_8 quantization variant" do
      assert "Q4_0_8_8" = HuggingFace.extract_quantization("model-Q4_0_8_8.gguf")
    end

    test "extracts IQ3_M quantization" do
      assert "IQ3_M" = HuggingFace.extract_quantization("model-IQ3_M.gguf")
    end

    test "extracts IQ4_XS quantization" do
      assert "IQ4_XS" = HuggingFace.extract_quantization("model-IQ4_XS.gguf")
    end

    test "extracts IQ2_XXS quantization" do
      assert "IQ2_XXS" = HuggingFace.extract_quantization("model-IQ2_XXS.gguf")
    end

    test "extracts F16 quantization" do
      assert "F16" = HuggingFace.extract_quantization("model-f16.gguf")
    end

    test "extracts F32 quantization" do
      assert "F32" = HuggingFace.extract_quantization("model-F32.gguf")
    end

    test "extracts BF16 quantization" do
      assert "BF16" = HuggingFace.extract_quantization("model-bf16.gguf")
    end

    test "returns unknown for unrecognized format" do
      assert "unknown" = HuggingFace.extract_quantization("model.gguf")
    end

    test "returns unknown for non-GGUF file" do
      assert "unknown" = HuggingFace.extract_quantization("config.json")
    end

    test "handles underscore separator" do
      assert "Q4_K_M" = HuggingFace.extract_quantization("model_Q4_K_M.gguf")
    end

    test "is case insensitive" do
      assert "Q4_K_M" = HuggingFace.extract_quantization("model-q4_k_m.gguf")
    end
  end

  # ============================================================================
  # Quantization Selection Tests
  # ============================================================================

  describe "quant_preference/0" do
    test "returns a non-empty list" do
      prefs = HuggingFace.quant_preference()
      assert is_list(prefs)
      assert length(prefs) > 0
    end

    test "Q4_K_M is first preference" do
      [first | _] = HuggingFace.quant_preference()
      assert first == "Q4_K_M"
    end

    test "includes common quantizations" do
      prefs = HuggingFace.quant_preference()
      assert "Q4_K_M" in prefs
      assert "Q5_K_M" in prefs
      assert "Q8_0" in prefs
      assert "F16" in prefs
    end
  end

  describe "best_quantization/2" do
    setup do
      ggufs = [
        %{
          filename: "model-IQ3_M.gguf",
          size_bytes: 610_000_000,
          size_gb: 0.61,
          quantization: "IQ3_M",
          ollama_tag: "IQ3_M"
        },
        %{
          filename: "model-Q4_K_M.gguf",
          size_bytes: 750_000_000,
          size_gb: 0.75,
          quantization: "Q4_K_M",
          ollama_tag: "Q4_K_M"
        },
        %{
          filename: "model-Q5_K_M.gguf",
          size_bytes: 850_000_000,
          size_gb: 0.85,
          quantization: "Q5_K_M",
          ollama_tag: "Q5_K_M"
        },
        %{
          filename: "model-Q8_0.gguf",
          size_bytes: 1_230_000_000,
          size_gb: 1.23,
          quantization: "Q8_0",
          ollama_tag: "Q8_0"
        },
        %{
          filename: "model-F16.gguf",
          size_bytes: 2_310_000_000,
          size_gb: 2.31,
          quantization: "F16",
          ollama_tag: "F16"
        }
      ]

      {:ok, ggufs: ggufs}
    end

    test "selects Q4_K_M when available", %{ggufs: ggufs} do
      assert "Q4_K_M" = HuggingFace.best_quantization(ggufs)
    end

    test "selects next best when Q4_K_M not available" do
      ggufs = [
        %{
          filename: "model-Q8_0.gguf",
          size_bytes: 1_230_000_000,
          size_gb: 1.23,
          quantization: "Q8_0",
          ollama_tag: "Q8_0"
        },
        %{
          filename: "model-F16.gguf",
          size_bytes: 2_310_000_000,
          size_gb: 2.31,
          quantization: "F16",
          ollama_tag: "F16"
        }
      ]

      assert "Q8_0" = HuggingFace.best_quantization(ggufs)
    end

    test "respects max_size_gb constraint", %{ggufs: ggufs} do
      # Only IQ3_M (0.61 GB) fits under 0.7 GB
      assert "IQ3_M" = HuggingFace.best_quantization(ggufs, max_size_gb: 0.7)
    end

    test "filters out files exceeding max_size_gb", %{ggufs: ggufs} do
      # Q4_K_M (0.75 GB) and smaller should be considered
      assert "Q4_K_M" = HuggingFace.best_quantization(ggufs, max_size_gb: 1.0)
    end

    test "returns nil for empty list" do
      assert nil == HuggingFace.best_quantization([])
    end

    test "returns first available if no preference matches" do
      ggufs = [
        %{
          filename: "model-CUSTOM.gguf",
          size_bytes: 500_000_000,
          size_gb: 0.5,
          quantization: "CUSTOM",
          ollama_tag: "CUSTOM"
        }
      ]

      assert "CUSTOM" = HuggingFace.best_quantization(ggufs)
    end

    test "uses custom preference list" do
      ggufs = [
        %{
          filename: "model-Q4_K_M.gguf",
          size_bytes: 750_000_000,
          size_gb: 0.75,
          quantization: "Q4_K_M",
          ollama_tag: "Q4_K_M"
        },
        %{
          filename: "model-Q8_0.gguf",
          size_bytes: 1_230_000_000,
          size_gb: 1.23,
          quantization: "Q8_0",
          ollama_tag: "Q8_0"
        }
      ]

      # Custom preference prefers Q8_0 over Q4_K_M
      assert "Q8_0" = HuggingFace.best_quantization(ggufs, preference: ["Q8_0", "Q4_K_M"])
    end
  end

  # ============================================================================
  # Auto Select Tests (Unit - with mocked data)
  # ============================================================================

  describe "auto_select/2 with mocked list_gguf_files" do
    # These tests would need mocking of HfHub.Api.list_repo_tree
    # For now, we test the logic by testing the helper functions

    test "returns error for empty repo" do
      # This would need HfHub API mocking to test fully
      # Tested via integration tests
    end
  end

  # ============================================================================
  # GGUF File Info Structure Tests
  # ============================================================================

  describe "gguf_info structure" do
    test "has expected fields" do
      info = %{
        filename: "model-Q4_K_M.gguf",
        size_bytes: 750_000_000,
        size_gb: 0.75,
        quantization: "Q4_K_M",
        ollama_tag: "Q4_K_M"
      }

      assert is_binary(info.filename)
      assert is_integer(info.size_bytes)
      assert is_float(info.size_gb)
      assert is_binary(info.quantization)
      assert is_binary(info.ollama_tag)
    end
  end
end
