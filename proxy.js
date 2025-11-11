import { NextResponse } from "next/server";

const TARGET = "https://www.truseamossanalyzer.one/Performance_Dashboard";

export default function proxy(request) {
  const upstream = new URL(TARGET);
  upstream.search = request.nextUrl.search;
  return NextResponse.rewrite(upstream);
}

export const config = {
  matcher: "/:path*",
};

