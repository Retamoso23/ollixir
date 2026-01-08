defmodule Ollixir.WebCloudTest do
  use ExUnit.Case, async: true

  @moduletag :cloud_api

  setup_all do
    key = System.get_env("OLLAMA_API_KEY")

    if key in [nil, ""] do
      flunk("OLLAMA_API_KEY is required to run cloud_api tests.")
    end

    {:ok, client: Ollixir.init()}
  end

  test "web_search returns results", %{client: client} do
    assert {:ok, %Ollixir.Web.SearchResponse{results: results}} =
             Ollixir.web_search(client, query: "elixir", max_results: 1)

    assert is_list(results)
  end

  test "web_fetch returns content", %{client: client} do
    assert {:ok, %Ollixir.Web.FetchResponse{content: content}} =
             Ollixir.web_fetch(client, url: "https://ollama.com")

    assert is_binary(content)
  end
end
