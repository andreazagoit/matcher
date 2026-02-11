import { NextResponse } from "next/server";
import { resetAllDailyMatches } from "@/lib/models/matches/operations";

export async function GET(request: Request) {
    const authHeader = request.headers.get("authorization");

    if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
        return new Response("Unauthorized", { status: 401 });
    }

    try {
        await resetAllDailyMatches();
        return NextResponse.json({ success: true, message: "Daily matches reset successfully" });
    } catch (error) {
        console.error("Cron reset error:", error);
        return NextResponse.json({ success: false, error: "Internal Server Error" }, { status: 500 });
    }
}
