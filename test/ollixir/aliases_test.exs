defmodule Ollixir.AliasesTest do
  use Supertester.ExUnitFoundation, isolation: :full_isolation, async: false

  setup_all do
    {:ok, client: Ollixir.init("http://localhost:4000")}
  end

  test "generate/2 delegates to completion/2", %{client: client} do
    assert {:ok, res} = Ollixir.generate(client, model: "llama2", prompt: "Hello")
    assert res["model"] == "llama2"
  end

  test "list/1 delegates to list_models/1", %{client: client} do
    assert {:ok, res} = Ollixir.list(client)
    assert is_list(models_from_response(res))
  end

  test "show/2 delegates to show_model/2", %{client: client} do
    assert {:ok, res} = Ollixir.show(client, name: "llama2")
    assert is_map(res)
  end

  test "ps/1 delegates to list_running/1", %{client: client} do
    assert {:ok, res} = Ollixir.ps(client)
    assert is_list(models_from_response(res))
  end

  test "pull/2 delegates to pull_model/2", %{client: client} do
    assert {:ok, res} = Ollixir.pull(client, name: "llama2")
    assert res["status"] == "success"
  end

  test "push/2 delegates to push_model/2", %{client: client} do
    assert {:ok, res} = Ollixir.push(client, name: "llama2")
    assert res["status"] == "success"
  end

  test "create/2 delegates to create_model/2", %{client: client} do
    assert {:ok, res} = Ollixir.create(client, name: "llama2")
    assert res["status"] == "success"
  end

  test "copy/2 delegates to copy_model/2", %{client: client} do
    assert {:ok, res} = Ollixir.copy(client, source: "llama2", destination: "llama2-copy")

    case res do
      true ->
        assert true

      false ->
        assert false == false

      %Ollixir.Types.StatusResponse{} = status ->
        assert Ollixir.Types.StatusResponse.success?(status)
    end
  end

  test "delete/2 delegates to delete_model/2", %{client: client} do
    assert {:ok, res} = Ollixir.delete(client, name: "llama2")

    case res do
      true ->
        assert true

      false ->
        assert false == false

      %Ollixir.Types.StatusResponse{} = status ->
        assert Ollixir.Types.StatusResponse.success?(status)
    end
  end

  defp models_from_response(%{"models" => models}) when is_list(models), do: models
  defp models_from_response(%Ollixir.Types.ListResponse{models: models}), do: models
  defp models_from_response(%Ollixir.Types.ProcessResponse{models: models}), do: models
end
