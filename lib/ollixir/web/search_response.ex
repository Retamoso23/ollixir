defmodule Ollixir.Web.SearchResponse do
  @moduledoc """
  Response from web_search.
  """

  use Ollixir.Types.Base

  alias Ollixir.Web.SearchResult

  @type t :: %__MODULE__{results: [SearchResult.t()]}
  defstruct results: []

  def from_map(%{"results" => results}) when is_list(results) do
    %__MODULE__{results: Enum.map(results, &SearchResult.from_map/1)}
  end

  def from_map(_), do: %__MODULE__{}
end
