import { NextRequest, NextResponse } from "next/server";

/**
 * Proxy upload requests to the FastAPI backend.
 * This fixes CORS issues when deploying to production.
 */
export async function POST(request: NextRequest) {
    try {
        // Get collection_id from URL params
        const { searchParams } = new URL(request.url);
        const collectionId = searchParams.get("collection_id") || "default";

        // Get the form data from the request
        const formData = await request.formData();

        // Get backend URL from environment or default to localhost
        const backendUrl = process.env.RAG_BACKEND_URL || "http://127.0.0.1:8000";

        // Forward the request to the FastAPI backend
        const response = await fetch(
            `${backendUrl}/upload?collection_id=${collectionId}`,
            {
                method: "POST",
                body: formData,
            }
        );

        // Check if the response is OK
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({
                detail: `Backend error: ${response.status}`,
            }));
            return NextResponse.json(
                { error: errorData.detail || "Upload failed" },
                { status: response.status }
            );
        }

        // Return the backend's response
        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error("Upload proxy error:", error);
        const message = error instanceof Error ? error.message : "Upload failed";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}
