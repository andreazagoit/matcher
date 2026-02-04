export async function graphql<T = any>(query: string, variables?: Record<string, any>): Promise<T> {
    const res = await fetch("/api/graphql", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ query, variables }),
    });

    const json = await res.json();

    if (json.errors) {
        throw new Error(json.errors[0].message);
    }

    return json.data;
}
